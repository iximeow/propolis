// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Mechanisms required to implement a block device

use std::any::Any;
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::task::{Context, Poll};
use std::time::Instant;

use pin_project_lite::pin_project;
use tokio::sync::futures::Notified;
use tokio::sync::Notify;

use crate::block::attachment::Bitmap;
use crate::block::{self, devq_id, probes, Operation, Request};
use crate::block::{DeviceId, MetricConsumer, QueueId, WorkerId};

/// Each emulated block device will have one or more [DeviceQueue]s which can be
/// polled through [next_req()](DeviceQueue::next_req()) to emit IO requests.
/// The completions for those requests are then processed through
/// [complete()](DeviceQueue::complete()) calls.
pub trait DeviceQueue: Send + Sync + 'static {
    /// Requests emitted from a [DeviceQueue] may require some associated state
    /// in order to communicate their completion to the guest.  The `Token` type
    /// represents that state.
    type Token: Send + Sync + 'static;

    /// Get the next [Request] (if any) from this queue.  Supporting data
    /// included with the request consists of the necessary [Self::Token] as
    /// well an optional [queued-time](Instant).
    fn next_req(&self) -> Option<(Request, Self::Token, Option<Instant>)>;

    /// Emit a completion for a processed request, identified by its
    /// [token](Self::Token).
    fn complete(
        &self,
        op: block::Operation,
        result: block::Result,
        token: Self::Token,
    );

    fn complete_bulk(&self, completions: BulkCompletions<Self::Token>) {
        for (op, result, token) in completions {
            self.complete(op, result, token)
        }
    }

    /// Explicitly abandon a queue token, never to be used for an I/O
    /// completion.
    ///
    /// A token's typical lifecycle is to be produced by
    /// [`DeviceQueue::next_req`], operated on, and completed with a result by
    /// [`DeviceQueue::complete`]. If the device's queues are dissociated, such
    /// as by a reset of the device, we may want to shortcut this lifecycle and
    /// destroy the token immediately.
    ///
    /// `DeviceQueue` implementations may use `Token`s that panic on `Drop`, to
    /// flag errnoenous discards of request tokens without completing them.
    /// `abandon`, instead, is an escape hatch in the case one genuinely must
    /// discard an I/O token without fulfilling the operation.
    fn abandon(&self, token: Self::Token);
}

pub struct BulkCompletions<T: Send + Sync + 'static> {
    items: std::vec::IntoIter<QmResult>,
    _phantom: PhantomData<T>,
}
impl<T: Send + Sync + 'static> Iterator for BulkCompletions<T> {
    type Item = (block::Operation, block::Result, T);

    fn next(&mut self) -> Option<Self::Item> {
        let QmResult { token, op, result, .. } = self.items.next()?;
        let token =
            token.downcast::<T>().expect("token should successfully downcast");
        Some((op, result, *token))
    }
}

/// A wrapper for an IO [Request] bearing necessary tracking information to
/// issue its completion back to the [queue](DeviceQueue) from which it came.
///
/// A panic will occur a `DeviceRequest` instance is dropped without calling
/// [complete()](DeviceRequest::complete()).
pub struct DeviceRequest {
    req: Request,
    id: ReqId,
    source: Weak<QueueMinder>,
    _nodrop: NoDropDevReq,
}
impl DeviceRequest {
    fn new(id: ReqId, req: Request, source: Weak<QueueMinder>) -> Self {
        Self { req, id, source, _nodrop: NoDropDevReq }
    }

    /// Get the underlying block [Request]
    pub fn req(&self) -> &Request {
        &self.req
    }

    /// Issue a completion for this [Request].
    pub fn complete(self, result: super::Result) {
        let DeviceRequest { id, source, _nodrop, .. } = self;
        std::mem::forget(_nodrop);

        if let Some(src) = source.upgrade() {
            src.complete(id, result);
        }
    }
}

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
struct Coalescer {
    /// The minder for which we're handling coalescing.
    minder: Arc<QueueMinder>,
    /// Notify to tell us that an I/O has been completed, and we should
    /// set a deadline at which point this I/O and any that have come in
    /// since should be flushed.
    start_coalescing: Notify,
    /// Notify to tell us that a batch of completions have been sent; if
    /// we were waiting to send an interrupt we can give up and go back
    /// to waiting for new I/Os.
    completions_sent: Notify,
    /// Nanos when the current batch's first completion was submitted.
    /// "0" is a magic value indicating that no completion is currently
    /// ongoing.
    batch_start: AtomicU64,
    destroyed: AtomicBool,
}

impl Coalescer {
    fn new(minder: Arc<QueueMinder>) -> Self {
        Self {
            minder,
            start_coalescing: Notify::new(),
            completions_sent: Notify::new(),
            batch_start: AtomicU64::new(0),
            destroyed: AtomicBool::new(false),
        }
    }

    fn should_coalesce(&self, op: Operation, existing: &[QmResult]) -> bool {
        // Arbitrary limits for now
        const COALESCE_MAX_COUNT: usize = 16;

        if op.is_flush() {
            return false;
        }
        if existing.len() >= COALESCE_MAX_COUNT {
            return false;
        }

        true
    }

    fn completed_io(&self) {
        let res = self.batch_start.compare_exchange(
            0,
            1,
            Ordering::SeqCst,
            Ordering::Relaxed,
        );
        if res.is_ok() {
            self.start_coalescing.notify_one();
        }
    }

    fn flush_completions(&self) {
        // Zero batch_start before flushing completions so that if an
        // I/O comes in while or after flushing completions it will
        // set `batch_start` and notify us that we should start
        // coalescing.
        //
        // It's possible that such a completion actually gets flushed in
        // *this* `flush_completions()` we're about to call. In that
        // case it's even possible that if no other completions arrive,
        // we'll wake when the deadline is reached and attempt to flush
        // zero completions.
        self.batch_start.store(0, Ordering::Release);
        let state = self.minder.state.lock().unwrap();
        if !state.coalesced.is_empty() {
            self.minder.flush_coalesced(state, None);
        }
    }

    fn shutdown(&self) {
        self.destroyed.store(true, Ordering::Release);
        // A task is likely waiting to be told to start notifying,
        // so kick it to let it see it's time to wrap up.
        self.start_coalescing.notify_one();
    }

    async fn coalesce_completions(&self) {
        loop {
            self.start_coalescing.notified().await;

            if self.destroyed.load(Ordering::Acquire) {
                println!("QueueMinder destroed its Coalescer");
                break;
            }

            let notify = self.completions_sent.notified();

            // arbitrary choice of 100 microsecond coalescing timeout.
            // TODO: a better number here would consider the number of
            // queues that might be interrupted and how many I/Os we'd
            // expect to actually accumulate inside 100us.
            const COALESCE_MAX_WAIT: Duration = Duration::from_micros(100);

            // TODO: only sleep by the difference between when the I/O
            // was completed and now, rather than 100us from now.
            // hopefully the difference should be tiny but tens or
            // twenties of microseconds have been seen sunk into waiting
            // on locks..
            let delay = tokio::time::sleep(COALESCE_MAX_WAIT);

            tokio::select! {
                _ = notify => {
                   // the completion batch is done, wait for another I/O to
                   // start coalescing
                   continue;
                },
                _ = delay => {
                    // aggregation timeout, send what we've seen
                    self.flush_completions();
                }
            }
        }
    }
}

/// Marker struct to ensure that [DeviceRequest] consumers call
/// [complete()](DeviceRequest::complete()), rather than silently dropping it.
struct NoDropDevReq;
impl Drop for NoDropDevReq {
    fn drop(&mut self) {
        panic!("DeviceRequest should be complete()-ed before drop");
    }
}

/// Closure to permit [QueueMinder] to type-erase the calling of
/// [DeviceQueue::next_req()].
type NextReqFn = Box<
    dyn Fn() -> Option<(
            block::Request,
            Box<dyn Any + Send + Sync>,
            Option<Instant>,
        )> + Send
        + Sync,
>;

/// Closure to permit [QueueMinder] to type-erase the calling of
/// [DeviceQueue::complete()].
type CompleteReqFn = Box<
    dyn Fn(Operation, block::Result, Box<dyn Any + Send + Sync>) + Send + Sync,
>;

/// Closure to permit [QueueMinder] to type-erase the calling of
/// [DeviceQueue::complete_bulk()].
type CompleteBulkFn = Box<dyn Fn(std::vec::IntoIter<QmResult>) + Send + Sync>;

/// Closure to permit [QueueMinder] to type-erase the calling of
/// [DeviceQueue::abandon()].
type AbandonReqFn = Box<dyn Fn(Box<dyn Any + Send + Sync>) + Send + Sync>;

struct QmRequest {
    token: Box<dyn Any + Send + Sync>,
    op: Operation,
    when_queued: Instant,
    when_started: Instant,
}
struct QmResult {
    token: Box<dyn Any + Send + Sync>,
    op: Operation,
    result: block::Result,
    when_processed: Instant,
}
impl QmResult {
    fn from_req(
        req: QmRequest,
        result: block::Result,
        when_processed: Instant,
    ) -> Self {
        Self { token: req.token, op: req.op, result, when_processed }
    }
}

struct QmInner {
    next_id: ReqId,
    /// Map of [WorkerId]s which we emitted [None] to via
    /// [QueueMinder::next_req()] and which are likely candidates to notify when
    /// this queue has new entries.
    notify_workers: Bitmap,
    paused: bool,
    /// Has this QueueMinder been destroyed? Since `QueueMinder` is typically in
    /// an Arc, `destroy` may be called while there are other references
    /// outstanding - concurrent completions that have just upgraded their
    /// weak refs, for example. When the minder has been "destroyed", those I/Os
    /// should gracefully abort.
    destroyed: bool,
    in_flight: BTreeMap<ReqId, QmRequest>,
    coalesced: Vec<QmResult>,
    metric_consumer: Option<Arc<dyn MetricConsumer>>,
    /// Number of [Request] completions which are currently being processed by
    /// the device.  This is tracked only for requests which are the last entry
    /// removed from `in_flight`, as a means providing accurate results from
    /// [NoneInFlight].
    processing_last: usize,
    /// Just enough state to manage time-based I/O completion flushes.
    /// TODO: would be nice to move *all* the coalescing here. Might need to
    /// give this a reference to `complete_bulk_fn` or something?
    coalescer: Option<Arc<Coalescer>>,
}
impl Default for QmInner {
    fn default() -> Self {
        Self {
            next_id: ReqId::START,
            notify_workers: Bitmap::default(),
            paused: false,
            destroyed: false,
            processing_last: 0,
            in_flight: BTreeMap::new(),
            coalesced: Vec::new(),
            metric_consumer: None,
            coalescer: None,
        }
    }
}

pub(super) struct QueueMinder {
    pub queue_id: QueueId,
    pub device_id: DeviceId,
    state: Mutex<QmInner>,
    self_ref: Weak<Self>,
    notify: Notify,
    /// Type-erased wrapper function for [DeviceQueue::next_req()]
    next_req_fn: NextReqFn,
    /// Type-erased wrapper function for [DeviceQueue::complete()]
    complete_req_fn: CompleteReqFn,
    /// Type-erased wrapper function for [DeviceQueue::complete_bulk()]
    complete_bulk_fn: CompleteBulkFn,
    /// Type-erased wrapper function for [DeviceQueue::abandon()]
    abandon_req_fn: AbandonReqFn,
}

impl QueueMinder {
    pub fn destroy(self: Arc<Self>) {
        // Up-front, it would be nice to assert that we have the last strong ref
        // on this `QueueMinder`. We might not actually though: the controller
        // may be reset at the same time we're completing I/Os, and those
        // completions have upgraded their ref back to the minder.
        //
        // So, do *not* `assert_eq!(Arc::strong_count(&self), 1);`.

        let mut state = self.state.lock().unwrap();

        // A minder can only be destroyed once. To destroy it more than once
        // would imply it was dissociated from a queue a second time, and for
        // that to happen the destroyed minder would have had to be associated
        // to a queue. Nonsense!
        assert!(!state.destroyed);
        state.destroyed = true;

        if let Some(coalescer) = state.coalescer.as_ref() {
            coalescer.shutdown();
        }

        if state.in_flight.len() > 0 {
            let old = std::mem::replace(&mut state.in_flight, BTreeMap::new());
            for (_, QmRequest { token, .. }) in old.into_iter() {
                (self.abandon_req_fn)(token);
            }
        }
        assert_eq!(state.in_flight.len(), 0);

        if state.coalesced.len() > 0 {
            let old = std::mem::replace(&mut state.coalesced, Vec::new());
            for QmResult { token, .. } in old.into_iter() {
                (self.abandon_req_fn)(token);
            }
        }
        assert_eq!(state.coalesced.len(), 0);
    }

    fn enable_coalescing(&self, coalescer: Arc<Coalescer>) {
        let mut state = self.state.lock().unwrap();
        assert!(!state.destroyed);
        state.coalescer = Some(coalescer);
    }

    pub fn new<DQ: DeviceQueue>(
        rt_handle: tokio::runtime::Handle,
        queue: Arc<DQ>,
        device_id: DeviceId,
        queue_id: QueueId,
    ) -> Arc<Self> {
        let device_queue_ref = queue.clone();
        let next_req_fn: NextReqFn = Box::new(move || {
            let (req, token, when_queued) = device_queue_ref.next_req()?;
            Some((
                req,
                Box::new(token) as Box<dyn Any + Send + Sync>,
                when_queued,
            ))
        });

        let device_queue_ref = queue.clone();
        let complete_req_fn: CompleteReqFn =
            Box::new(move |op, result, token| {
                let token = token
                    .downcast::<DQ::Token>()
                    .expect("token type unchanged");
                let token = *token;
                device_queue_ref.complete(op, result, token);
            });

        let device_queue_ref = queue.clone();
        let abandon_req_fn: AbandonReqFn = Box::new(move |token| {
            let token =
                token.downcast::<DQ::Token>().expect("token type unchanged");
            let token = *token;
            device_queue_ref.abandon(token);
        });

        let complete_bulk_fn: CompleteBulkFn = Box::new(move |items| {
            let bulk = BulkCompletions { items, _phantom: PhantomData };
            queue.complete_bulk(bulk);
        });

        let minder_ref = Arc::new_cyclic(|self_ref| Self {
            queue_id,
            device_id,
            state: Mutex::new(QmInner::default()),
            self_ref: self_ref.clone(),
            notify: Notify::new(),
            next_req_fn,
            complete_req_fn,
            complete_bulk_fn,
            abandon_req_fn,
        });

        let minder = Arc::clone(&minder_ref);
        let coalescer = Arc::new(Coalescer::new(minder));

        let coalescer_ref = Arc::clone(&coalescer);
        let _coalescer = rt_handle.spawn(async move {
            coalescer_ref.coalesce_completions().await;
        });

        let minder = Arc::clone(&minder_ref);
        minder.enable_coalescing(coalescer);

        minder
    }

    /// Attempt to fetch the next IO request from this queue for a worker.
    ///
    /// If no requests are available, that worker (specified by `wid`) will be
    /// recorded so it can be notified if/when the guest notifies this queue
    /// that more requests are available.
    pub fn next_req(&self, wid: WorkerId) -> Option<DeviceRequest> {
        let mut state = self.state.lock().unwrap();
        if state.destroyed {
            return None;
        }
        if state.paused {
            state.notify_workers.set(wid);
            return None;
        }
        if let Some((req, token, when_queued)) = (self.next_req_fn)() {
            let id = state.next_id;
            state.next_id.advance();

            let devqid = devq_id(self.device_id, self.queue_id);
            match req.op {
                Operation::Read(off, len) => {
                    probes::block_begin_read!(|| {
                        (devqid, id, off as u64, len as u64)
                    });
                }
                Operation::Write(off, len) => {
                    probes::block_begin_write!(|| {
                        (devqid, id, off as u64, len as u64)
                    });
                }
                Operation::Flush => {
                    probes::block_begin_flush!(|| { (devqid, id) });
                }
                Operation::Discard(off, len) => {
                    probes::block_begin_discard!(|| {
                        (devqid, id, off as u64, len as u64)
                    });
                }
            }
            let when_started = Instant::now();
            let old = state.in_flight.insert(
                id,
                QmRequest {
                    token,
                    op: req.op,
                    when_queued: when_queued.unwrap_or(when_started),
                    when_started,
                },
            );
            assert!(old.is_none(), "request IDs should not overlap");

            Some(DeviceRequest::new(id, req, self.self_ref.clone()))
        } else {
            state.notify_workers.set(wid);
            None
        }
    }

    /// Process a completion for an in-flight IO request on this queue.
    pub fn complete(&self, id: ReqId, result: block::Result) {
        let mut state = self.state.lock().unwrap();
        let Some(ent) = state.in_flight.remove(&id) else {
            // If we lost state for this I/O, we better have gotten here because
            // the controller was reset and dissociated all queues. In that case
            // we should have destroyed the `QueueMinder`s, so assert that is
            // the case.
            assert!(state.destroyed);

            // One must imagine the guest would be happy to know the I/O *was*
            // completed after all, but we can no longer do anything about it.
            // We don't even know when it started anymore, so we can't report
            // meaningful metrics about it.
            return;
        };
        let metric_consumer = state.metric_consumer.as_ref().map(Arc::clone);

        let when_processed = Instant::now();
        let time_queued = ent.when_started.duration_since(ent.when_queued);
        let time_processed = when_processed.duration_since(ent.when_started);

        let ns_queued = time_queued.as_nanos() as u64;
        let ns_processed = time_processed.as_nanos() as u64;
        let rescode = result as u8;
        let devqid = devq_id(self.device_id, self.queue_id);
        match ent.op {
            Operation::Read(..) => {
                probes::block_complete_read!(|| {
                    (devqid, id, rescode, ns_processed, ns_queued)
                });
            }
            Operation::Write(..) => {
                probes::block_complete_write!(|| {
                    (devqid, id, rescode, ns_processed, ns_queued)
                });
            }
            Operation::Flush => {
                probes::block_complete_flush!(|| {
                    (devqid, id, rescode, ns_processed, ns_queued)
                });
            }
            Operation::Discard(..) => {
                probes::block_complete_discard!(|| {
                    (devqid, id, rescode, ns_processed, ns_queued)
                });
            }
        }

        let op = ent.op;
        let qm_result = QmResult::from_req(ent, result, when_processed);
        if let Some(coalescer) = state.coalescer.as_ref() {
            if coalescer.should_coalesce(op, &state.coalesced) {
                // queue for a later coalesced completion
                coalescer.completed_io();
                state.coalesced.push(qm_result);
            } else {
                // deliver immediately completion (along with any others which have
                // been coalesced)
                self.flush_coalesced(state, Some(qm_result));
            }
        } else {
            let QmResult { token, op, result, when_processed } = qm_result;
            (self.complete_req_fn)(op, result, token);

            probes::block_completion_single!(|| {
                (devqid, when_processed.elapsed().as_nanos() as u64)
            });
        }

        // Report the completion to the metrics consumer, if one exists
        if let Some(consumer) = metric_consumer {
            consumer.request_completed(
                self.queue_id,
                op,
                result,
                time_queued,
                time_processed,
            );
        }
    }

    fn flush_coalesced(
        &self,
        mut state: MutexGuard<QmInner>,
        latest: Option<QmResult>,
    ) {
        let devqid = devq_id(self.device_id, self.queue_id);
        let is_last_req = state.in_flight.is_empty();
        if is_last_req {
            state.processing_last += 1;
        }
        if let Some(first_req_done) =
            state.coalesced.get(0).map(|qr| qr.when_processed)
        {
            let mut items = std::mem::replace(&mut state.coalesced, Vec::new());
            if let Some(ent) = latest {
                items.push(ent);
            }
            let count = items.len();

            drop(state);
            (self.complete_bulk_fn)(items.into_iter());

            probes::block_completion_coalesced!(|| {
                (devqid, first_req_done.elapsed().as_nanos() as u64, count)
            });
        } else {
            // No coalesced entries to flush
            //
            // Deliver completion for latest entry, if passed in
            drop(state);
            if let Some(ent) = latest {
                let QmResult { token, op, result, when_processed } = ent;
                (self.complete_req_fn)(op, result, token);

                probes::block_completion_single!(|| {
                    (devqid, when_processed.elapsed().as_nanos() as u64)
                });
            }
        }

        // We must track how many completions are being processed by the device,
        // since they are done outside the state lock, in order to present a
        // reliably accurate accurate signal of when the device has no more
        // in-flight requests.
        if is_last_req {
            let mut state = self.state.lock().unwrap();
            state.processing_last =
                state.processing_last.checked_sub(1).expect("dont underflow");
            if state.in_flight.is_empty() && state.processing_last == 0 {
                self.notify.notify_waiters();
            }
        }
    }

    /// Take the bitmap of the workers which should be notified that this queue
    /// may now have requests available.
    ///
    /// Bits in this map correspond to workers that either should be
    /// [`WorkerSlot::wake`]'d or returned to this `QueueMinder` via
    /// [`add_notifications`]. Failure to do so will result in idle workers
    /// never being woken for future work.
    pub(in crate::block) fn take_notifications(&self) -> Option<Bitmap> {
        let mut state = self.state.lock().unwrap();
        if state.paused {
            state.notify_workers = Bitmap::ALL;
            None
        } else {
            Some(state.notify_workers.take())
        }
    }

    /// Add a set of workers to be notified when this queue may have requests
    /// available.
    ///
    /// This should only be called with the remaining parts of a bitmap obtained
    /// from an ealier [`take_notifications`]. Using other bit patterns may
    /// result in wakeups to out-of-range worker IDs and subsequent panic.
    pub(in crate::block) fn add_notifications(&self, worker_ids: Bitmap) {
        let mut state = self.state.lock().unwrap();

        state.notify_workers.set_all(worker_ids);
    }

    /// Associate a [MetricConsumer] with this queue.
    ///
    /// It will be notified about each IO completion as they occur.
    pub(crate) fn set_metric_consumer(
        &self,
        consumer: Arc<dyn MetricConsumer>,
    ) {
        self.state.lock().unwrap().metric_consumer = Some(consumer);
    }

    pub(crate) fn pause(&self) {
        let mut state = self.state.lock().unwrap();
        state.paused = true;
        self.notify.notify_waiters();
    }

    pub(crate) fn resume(&self) {
        let mut state = self.state.lock().unwrap();
        state.paused = false;
        self.notify.notify_waiters();
    }

    pub(crate) fn none_in_flight(&self) -> NoneInFlight<'_> {
        NoneInFlight { minder: self, wait: self.notify.notified() }
    }
}

pin_project! {
    /// A [Future] which resolves to [Ready](Poll::Ready) when there are no
    /// requests being processed by an attached backend.
    pub(crate) struct NoneInFlight<'a> {
        minder: &'a QueueMinder,
        #[pin]
        wait: Notified<'a>
    }
}
impl Future for NoneInFlight<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();

        loop {
            let state = this.minder.state.lock().unwrap();
            if state.in_flight.is_empty() && state.processing_last == 0 {
                return Poll::Ready(());
            }
            // Keep the minder `state` lock held while polling the Notified
            // instance.  While it may not be strictly necessary, it matches the
            // conventions we expect from similar sync primitives such as CVs.
            if let Poll::Ready(_) = Notified::poll(this.wait.as_mut(), cx) {
                // Refresh fused future from Notify
                this.wait.set(this.minder.notify.notified());
            } else {
                return Poll::Pending;
            }
        }
    }
}

/// Unique ID assigned to a given block [Request].
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ReqId(u64);
impl ReqId {
    const START: Self = ReqId(0);

    fn advance(&mut self) {
        self.0 += 1;
    }
}
impl Borrow<u64> for ReqId {
    fn borrow(&self) -> &u64 {
        &self.0
    }
}
impl From<ReqId> for u64 {
    fn from(value: ReqId) -> Self {
        value.0
    }
}
