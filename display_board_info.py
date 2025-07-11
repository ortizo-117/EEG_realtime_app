from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import logging
import argparse
import time

def main():

    print("This is working?")
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=0)
    parser.add_argument('--ip-port', type=int, default=0)
    parser.add_argument('--ip-protocol', type=int, default=0)
    parser.add_argument('--ip-address', type=str, default='')
    parser.add_argument('--serial-port', type=str, default='')
    parser.add_argument('--mac-address', type=str, default='')
    parser.add_argument('--other-info', type=str, default='')
    parser.add_argument('--streamer-params', type=str, default='')
    parser.add_argument('--serial-number', type=str, default='')
    parser.add_argument('--board-id', type=int, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--master-board', type=int, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board_shim = BoardShim(args.board_id, params)
    
    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        print("Waiting to collect some data...")
        time.sleep(2)  # Give it time to buffer some data

        data = board_shim.get_current_board_data(256)  # Up to 256 most recent samples
        print(f"Data shape: {data.shape} (channels x samples)")

        # Print all available channels
        print("EEG Channels:", BoardShim.get_exg_channels(args.board_id))
        print("Auxiliary Channels:", BoardShim.get_analog_channels(args.board_id))
        print("Other Channels (e.g., digital):", BoardShim.get_other_channels(args.board_id))
        print("Timestamp Channel:", BoardShim.get_timestamp_channel(args.board_id))


    except Exception:
        logging.warning('Exception occurred', exc_info=True)

    finally:
        if board_shim.is_prepared():
            board_shim.stop_stream()
            board_shim.release_session()
            print("Stream stopped and session released.")

if __name__ == '__main__':
    main()