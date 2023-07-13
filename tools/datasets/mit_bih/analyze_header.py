import os


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    channel_num = None
    fps = None
    total_num = None
    channels = set()
    for file in os.listdir(data_path):
        if file.endswith(".hea"):
            with open(os.path.join(data_path, file), "r") as f:
                lines = f.readlines()
                _, cur_channel_num, cur_fps, cur_total_num = [
                    int(x) for x in lines[0].split()
                ]

                if channel_num is None:
                    channel_num = cur_channel_num
                else:
                    assert (
                        channel_num == cur_channel_num
                    ), f"channel_num: {channel_num}, cur_channel_num: {cur_channel_num} in {file}"

                if fps is None:
                    fps = cur_fps
                else:
                    assert fps == cur_fps, f"fps: {fps}, cur_fps: {cur_fps} in {file}"

                if total_num is None:
                    total_num = cur_total_num
                else:
                    assert (
                        total_num == cur_total_num
                    ), f"total_num: {total_num}, cur_total_num: {cur_total_num} in {file}"

                for i in range(cur_channel_num):
                    channels.add(lines[1 + i].split()[-1])
    print(f"channel_num: {channel_num}, fps: {fps}, total_num: {total_num}")
    print(f"channels: {sorted(channels)}")


if __name__ == "__main__":
    main()
