// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Tasks for vCPU backing threads and controls for them.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use propolis::{
    accessors::MemAccessor,
    bhyve_api,
    exits::{self, SuspendDetail, VmExitKind},
    pio::PioBus,
    vcpu::Vcpu,
    VmEntry,
};
use slog::{debug, error, info};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VcpuTaskError {
    #[error("Failed to spawn a vCPU backing thread: {0}")]
    BackingThreadSpawnFailed(std::io::Error),
}

pub struct VcpuTasks {
    tasks: Vec<(propolis::tasks::TaskCtrl, std::thread::JoinHandle<()>)>,
    generation: Arc<AtomicUsize>,
}

#[cfg_attr(test, mockall::automock)]
pub(crate) trait VcpuTaskController: Send + Sync + 'static {
    fn new_generation(&self);
    fn pause_all(&mut self);
    fn resume_all(&mut self);
    fn exit_all(&mut self);
}

/// A small struct to smuggle bits of `Machine` into `vcpu_loop`.
///
/// It'd be more "normal" to have an `Arc<Machine>` here, but in
/// `propolis-server` the `Vcpu` are set up as part of getting to a `Machine`.
/// Which makes it hard to have an `Arc<Machine>` this early. Instead, hold on
/// to the few parts of `Machine` we'll need for VM exit handling later.
///
/// This should be OK because vCPU tasks live and die with the `Machine`. We
/// won't have to "re-attach" a vCPU task to a new Machine, for example, so
/// tightly binding with no provision for updating attachments later is fine
/// just this once.
struct VcpuMachineState {
    bus_pio: Arc<PioBus>,
    acc_mem: MemAccessor,
}

impl VcpuTasks {
    pub(crate) fn new(
        machine: &propolis::Machine,
        event_handler: Arc<dyn super::vm::guest_event::VcpuEventHandler>,
        log: slog::Logger,
    ) -> Result<Self, VcpuTaskError> {
        let generation = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for vcpu in machine.vcpus.iter().map(Arc::clone) {
            let (task, ctrl) =
                propolis::tasks::TaskHdl::new_held(Some(vcpu.barrier_fn()));
            let task_log = log.new(slog::o!("vcpu" => vcpu.id));
            let task_event_handler = event_handler.clone();
            let task_gen = generation.clone();

            // I don't *think* vCPU structs outlive the Machine they're a part
            // of (unlike devices etc, which do outlive kernel VMMs since they
            // have interesting migration state etc).  So just attach to the
            // machine's MemAccessor here and have no provision for re-attaching
            // to some subsequent Machine instance.
            let acc_mem = MemAccessor::new_orphan();
            machine.acc_mem.adopt(&acc_mem, None);
            let state = VcpuMachineState {
                bus_pio: Arc::clone(&machine.bus_pio),
                acc_mem: acc_mem,
            };

            let thread = std::thread::Builder::new()
                .name(format!("vcpu-{}", vcpu.id))
                .spawn(move || {
                    Self::vcpu_loop(
                        vcpu.as_ref(),
                        state,
                        task,
                        task_event_handler,
                        task_gen,
                        task_log,
                    )
                })
                .map_err(VcpuTaskError::BackingThreadSpawnFailed)?;
            tasks.push((ctrl, thread));
        }

        Ok(Self { tasks, generation })
    }

    fn vcpu_loop(
        vcpu: &Vcpu,
        state: VcpuMachineState,
        task: propolis::tasks::TaskHdl,
        event_handler: Arc<dyn super::vm::guest_event::VcpuEventHandler>,
        generation: Arc<AtomicUsize>,

        log: slog::Logger,
    ) {
        info!(log, "Starting vCPU thread");
        let mut entry = VmEntry::Run;
        let mut exit = propolis::exits::VmExit::default();
        let mut local_gen = 0;
        loop {
            use propolis::tasks::Event;

            let mut force_exit_when_consistent = false;
            match task.pending_event() {
                Some(Event::Hold) => {
                    if !exit.kind.is_consistent() {
                        // Before the vCPU task can enter the held state, its
                        // associated in-kernel state must be driven to a point
                        // where it is consistent.
                        force_exit_when_consistent = true;
                    } else {
                        info!(log, "vCPU paused");
                        task.hold();
                        info!(log, "vCPU released from hold");

                        // If the VM was reset while the CPU was paused, clear out
                        // any re-entry reasons from the exit that occurred prior to
                        // the pause.
                        let current_gen = generation.load(Ordering::Acquire);
                        if local_gen != current_gen {
                            entry = VmEntry::Run;
                            local_gen = current_gen;
                        }

                        // This hold might have been satisfied by a request for the
                        // CPU to exit. Check for other pending events before
                        // re-entering the guest.
                        continue;
                    }
                }
                Some(Event::Exit) => break,
                None => {}
            }

            exit = match vcpu.run(&entry, force_exit_when_consistent) {
                Err(e) => {
                    event_handler.io_error_event(vcpu.id, e);
                    entry = VmEntry::Run;
                    continue;
                }
                Ok(exit) => exit,
            };

            entry = vcpu.process_vmexit(&exit).unwrap_or_else(|| {
                match exit.kind {
                    VmExitKind::Inout(pio) => {
                        debug!(&log, "Unhandled pio {:x?}", pio;
                                       "rip" => exit.rip);
                        VmEntry::InoutFulfill(exits::InoutRes::emulate_failed(
                            &pio,
                        ))
                    }
                    VmExitKind::Mmio(mmio) => {
                        debug!(&log, "Unhandled mmio {:x?}", mmio;
                                       "rip" => exit.rip);
                        VmEntry::MmioFulfill(exits::MmioRes::emulate_failed(
                            &mmio,
                        ))
                    }
                    VmExitKind::Rdmsr(msr) => {
                        debug!(&log, "Unhandled rdmsr {:08x}", msr;
                                       "rip" => exit.rip);
                        let _ = vcpu.set_reg(
                            bhyve_api::vm_reg_name::VM_REG_GUEST_RAX,
                            0,
                        );
                        let _ = vcpu.set_reg(
                            bhyve_api::vm_reg_name::VM_REG_GUEST_RDX,
                            0,
                        );
                        VmEntry::Run
                    }
                    VmExitKind::Wrmsr(msr, val) => {
                        debug!(&log, "Unhandled wrmsr {:08x} <- {:08x}", msr, val;
                                       "rip" => exit.rip);
                        VmEntry::Run
                    }
                    VmExitKind::Suspended(SuspendDetail { kind, when }) => {
                        use propolis::vcpu::Diagnostics;
                        match kind {
                            exits::Suspend::Halt => {
                                event_handler.suspend_halt_event(when);
                            }
                            exits::Suspend::Reset => {
                                event_handler.suspend_reset_event(when);
                            }
                            exits::Suspend::TripleFault(vcpuid) => {
                                slog::info!(
                                    &log,
                                    "triple fault on vcpu {}",
                                    vcpu.id;
                                    "state" => %Diagnostics::capture(vcpu)
                                );

                                if vcpuid == -1 || vcpuid == vcpu.id {
                                    event_handler
                                        .suspend_triple_fault_event(vcpu.id, when);
                                }
                            }
                        }

                        // This vCPU will not successfully re-enter the guest
                        // until the state worker does something about the
                        // suspend condition, so hold the task until it does so.
                        // Note that this blocks the task immediately.
                        //
                        // N.B.
                        // This usage assumes that it is safe for the VM
                        // controller to ask the task to hold again (which may
                        // occur if a separate pausing event is serviced in
                        // parallel on the state worker).
                        task.force_hold();
                        VmEntry::Run
                    }
                    VmExitKind::InstEmul(inst) => {
                        if &inst.inst_data[..2] == &[0xf3, 0x6c] {
                            // rep ins dx, byte [rdi]
                            let rcx = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RCX).unwrap();
                            let rdx = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RDX).unwrap();
                            let rdi = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RDI).unwrap();

                            let memctx = state.acc_mem.access().unwrap();
                            // if you want to put more than 16k over serial at once what is wrong
                            // with you.
                            assert!(rcx < 0x4000);
                            let mut buf = vec![0u8; rcx as usize];

                            for i in 0..(rcx as usize) {
                                buf[i] = state
                                    .bus_pio
                                    .handle_in(rdx as u16, 1)
                                    .expect("can port i/o") as u8;
                            }

                            use propolis::common::GuestAddr;
                            use propolis::common::GuestData;
                            memctx.write_from(
                                GuestAddr(rdi),
                                &mut GuestData::from(buf.as_mut_slice()),
                                rcx as usize
                            ).expect("ok");

                            vcpu.set_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RDI, rdi + rcx).unwrap();
                            vcpu.set_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RCX, 0).unwrap();

                            VmEntry::Run
                        } else if &inst.inst_data[..2] == &[0xf3, 0x6e] {
                            // rep outs dl, byte [rsi]
                            let rcx = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RCX).unwrap();
                            let rdx = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RDX).unwrap();
                            let rsi = vcpu.get_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RSI).unwrap();

                            let memctx = state.acc_mem.access().unwrap();
                            // if you want to put more than 16k over serial at once what is wrong
                            // with you.
                            assert!(rcx < 0x4000);
                            let mut buf = vec![0u8; rcx as usize];
                            use propolis::common::GuestAddr;
                            use propolis::common::GuestData;
                            memctx.read_into(
                                GuestAddr(rsi),
                                &mut GuestData::from(buf.as_mut_slice()),
                                rcx as usize
                            ).expect("ok");

                            for i in 0..(rcx as usize) {
                                state
                                    .bus_pio
                                    .handle_out(rdx as u16, 1, buf[i] as u32)
                                    .expect("can port i/o");
                            }
                            vcpu.set_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RSI, rsi + rcx).unwrap();
                            vcpu.set_reg(bhyve_api::vm_reg_name::VM_REG_GUEST_RCX, 0).unwrap();

                            VmEntry::Run
                        } else {
                            let diag = propolis::vcpu::Diagnostics::capture(vcpu);
                            error!(log,
                                   "instruction emulation exit on vCPU {}",
                                   vcpu.id;
                                   "context" => ?inst,
                                   "vcpu_state" => %diag);

                            event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                            VmEntry::Run
                        }
                    }
                    VmExitKind::Unknown(code) => {
                        error!(log,
                               "unrecognized exit code on vCPU {}",
                               vcpu.id;
                               "code" => code);

                        event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                        VmEntry::Run
                    }
                    // Bhyve emits the `Bogus` exit kind when there is no actual
                    // guest exit for user space to handle, but circumstances
                    // nevertheless dictate that the kernel VMM should exit to
                    // user space (e.g. a caller requested that all vCPUs be
                    // forced to exit to user space so their threads can
                    // rendezvous there).
                    //
                    // `process_vmexit` should always successfully handle this
                    // exit, since it never entails any work that could fail to
                    // be completed.
                    VmExitKind::Bogus => {
                        unreachable!(
                            "propolis-lib always handles VmExitKind::Bogus"
                        );
                    }
                    VmExitKind::Debug => {
                        error!(log,
                               "lib returned debug exit from vCPU {}",
                               vcpu.id);

                        event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                        VmEntry::Run
                    }
                    VmExitKind::VmxError(detail) => {
                        error!(log,
                               "unclassified VMX exit on vCPU {}",
                               vcpu.id;
                               "detail" => ?detail);

                        event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                        VmEntry::Run
                    }
                    VmExitKind::SvmError(detail) => {
                        error!(log,
                               "unclassified SVM exit on vCPU {}",
                               vcpu.id;
                               "detail" => ?detail);

                        event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                        VmEntry::Run
                    }
                    VmExitKind::Paging(gpa, fault_type) => {
                        let diag = propolis::vcpu::Diagnostics::capture(vcpu);
                        error!(log,
                               "unhandled paging exit on vCPU {}",
                               vcpu.id;
                               "gpa" => gpa,
                               "fault_type" => fault_type,
                               "vcpu_state" => %diag);

                        event_handler.unhandled_vm_exit(vcpu.id, exit.kind);
                        VmEntry::Run
                    }
                }
            });
        }
        info!(log, "Exiting vCPU thread for CPU {}", vcpu.id);
    }
}

impl VcpuTaskController for VcpuTasks {
    fn pause_all(&mut self) {
        for task in self.tasks.iter_mut().map(|t| &mut t.0) {
            task.hold().unwrap();
        }
    }

    fn new_generation(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn resume_all(&mut self) {
        for task in self.tasks.iter_mut().map(|t| &mut t.0) {
            task.run().unwrap();
        }
    }

    fn exit_all(&mut self) {
        for task in self.tasks.iter_mut().map(|t| &mut t.0) {
            task.exit();
        }

        for thread in self.tasks.drain(..) {
            thread.1.join().unwrap();
        }
    }
}
