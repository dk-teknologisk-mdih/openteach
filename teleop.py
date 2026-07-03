import time

import hydra
from openteach.components import TeleOperator

# Grace period (seconds) to let child processes handle Ctrl+C themselves
# (e.g. the Franka operator saves its recorded action history on shutdown)
# before we force-terminate them.
SHUTDOWN_GRACE_PERIOD = 15

@hydra.main(version_base = '1.2', config_path = '../config', config_name = 'teleop')
def main(configs):
    teleop = TeleOperator(configs)
    processes = teleop.get_processes()

    try:
        for process in processes:
            process.start()

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print('\nShutting down teleoperation...')
        # Ctrl+C (SIGINT) is delivered to every child process too, so give
        # them time to run their own KeyboardInterrupt handlers (and finish
        # writing pickle files) instead of killing them immediately.
        deadline = time.time() + SHUTDOWN_GRACE_PERIOD
        for process in processes:
            remaining = max(0.0, deadline - time.time())
            process.join(timeout = remaining)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()

        for process in processes:
            process.join()

if __name__ == '__main__':
    main()