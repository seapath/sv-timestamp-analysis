# Copyright (C) 2024, RTE (http://www.rte-france.com)
# Copyright (C) 2024 Savoir-faire Linux, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np

GREEN_COLOR = "#90EE90"
RED_COLOR = "#F08080"

def extract_sv(sv_file_path, streams):
    stream_number = 0
    with open(f"{sv_file_path}", "r", encoding="utf-8") as sv_file:
        sv_content = sv_file.read().splitlines()

    sv_id = np.array([str(item.split(":")[1]) for item in sv_content])
    stream_names = np.unique(sv_id)

    sv = [i for i in range(len(streams))]

    sv_it = np.array([str(item.split(":")[0]) for item in sv_content])
    sv_cnt = np.array([int(item.split(":")[2]) for item in sv_content])
    sv_timestamps = np.array([int(item.split(":")[3]) for item in sv_content])

    for stream in streams:
        try:
            id_occurrences = np.where(sv_id == stream_names[stream])
        except IndexError as e:
            print(f"Fatal: couldn't extract SV streams; is the -S argument correct? ({e})")
            exit(1)

        sv_it_occurrences = sv_it[id_occurrences]
        sv_cnt_occurrences = sv_cnt[id_occurrences]
        sv_timestamps_occurrences = sv_timestamps[id_occurrences]

        sv[stream_number] = [sv_it_occurrences, sv_cnt_occurrences, sv_timestamps_occurrences]

        stream_number += 1

    return sv

def detect_sv_drop(pub_sv, sub_sv, iteration_size=4000):
# This function is used to detect if there are any missed SV's in
# subscriber data, by testing the continuity of the SV counter of
# subscriber data.

    total_sv_drops = 0
    pub_sv_iter = np.sort(np.unique(pub_sv[0].astype(int)))

    for iteration in range(len(pub_sv_iter)):
        sub_sv_current_iter = np.where(sub_sv[0].astype(int) == iteration)[0]
        sub_sv_start_index = sub_sv_current_iter[0]
        sub_sv_end_index = sub_sv_current_iter[-1]+1
        sub_sv_cnt = sub_sv[1][sub_sv_start_index:sub_sv_end_index]

        diffs = np.diff(sub_sv_cnt) - 1
        neg_diffs = np.where(diffs < 0)[0]

        if neg_diffs.size > 0:
            print("Fatal: SV disordered detected")
            exit(1)

        if iteration_size-sub_sv_cnt[-1] > 0:
            diffs[-1] = iteration_size-sub_sv_cnt[-1] - 1
        if sub_sv_cnt[0] > 0:
            diffs[0] = sub_sv_cnt[0] - 1

        discontinuities = np.where(diffs > 0)[0]

        for disc in discontinuities:
            num_lost_values = diffs[disc]

            if num_lost_values == diffs[-1]:
                disc += 1
            if disc == 0:
                disc += -1
                num_lost_values += 1
            for _ in range(num_lost_values):
                pub_sv[0] = np.delete(pub_sv[0], sub_sv_start_index + disc + 1 )
                pub_sv[1] = np.delete(pub_sv[1], sub_sv_start_index + disc + 1 )
                pub_sv[2] = np.delete(pub_sv[2], sub_sv_start_index + disc + 1 )
            total_sv_drops += num_lost_values

    return total_sv_drops

def investigate_array_differences(array1, array2):
    # This function checks if pub and sub counter are well aligned.
    len1 = len(array1)
    len2 = len(array2)

    min_len = min(len1, len2)
    max_len = max(len1, len2)

    diff_indices = np.where(array1[:min_len] != array2[:min_len])[0]
    diffs = [(i, array1[i], array2[i]) for i in diff_indices]

    if len1 > len2:
        extra_elements = array1[len2:max_len]
        extra_info = {'array': 'array1', 'indices': np.arange(len2, max_len), 'values': extra_elements}
    elif len2 > len1:
        extra_elements = array2[len1:max_len]
        extra_info = {'array': 'array2', 'indices': np.arange(len1, max_len), 'values': extra_elements}
    else:
        extra_info = None

    return diffs, extra_info

def compute_latency(pub_sv, sub_sv):
    latencies = [[0]] * len(pub_sv)
    sv_drop = 0
    for stream in range(0, len(pub_sv)):
        if len(pub_sv[stream][1]) != len(sub_sv[stream][1]):
            sv_drop = detect_sv_drop(pub_sv[stream], sub_sv[stream])
            diffs, extra_info = investigate_array_differences(pub_sv[stream][1], sub_sv[stream][1])

            if diffs:
                print("Warning: SV counter misalignment between pub and sub")
            if extra_info:
                print(f"Warning: Extra elements in {extra_info['array']} at indices {extra_info['indices']}: {extra_info['values']}")
        latencies[stream] = sub_sv[stream][2] - pub_sv[stream][2]

    return latencies, sv_drop

def compute_pacing(sv):
    streams = len(sv)
    pacing = [[0]] * len(sv)
    for stream in range(0, streams):
        pacing[stream] = np.diff(sv[stream][2])
    return pacing

def get_stream_count(pub_sv):
    return np.unique(pub_sv).size

def compute_min(values):
    return np.min(values) if values.size > 0 else None

def compute_max(values):
    return np.max(values) if values.size > 0 else None

def compute_average(values):
    return np.round(np.mean(values)) if values.size > 0 else None

def compute_neglat(values):
    return np.count_nonzero(values < 0)

def save_latency_histogram(plot_type, values, streams, sub_name, output, subscriber):

    for stream in range(0, len(streams)):
        plt.hist(values[stream], bins=20, alpha=0.7)

        plt.xlabel(f"{plot_type} (us)")
        plt.ylabel("Occurrences")
        plt.yscale('log')
        plt.title(f"{sub_name} {plot_type} Histogram")

        filename = f"histogram_{sub_name}_stream_{stream}_{plot_type}_{subscriber}.png"
        filepath = os.path.realpath(f"{output}/results/{filename}")
        plt.savefig(filepath)
        print(f"Histogram saved as {filename}.")
        plt.close()

    return filepath

def generate_adoc(pub, hyp, sub, streams, hyp_name, sub_name, output, ttot):
    if not os.path.exists(f'{output}/results'):
        os.makedirs(f'{output}/results')

    with open(f"{output}/results/latency_tests.adoc", "w", encoding="utf-8") as adoc_file:
        subcriber_lines = textwrap.dedent(
            """
            ===== Subscriber {_subscriber_name_}
            {{set:cellbgcolor!}}
            |===
            |Number of IEC61850 Sampled Value |Minimum latency |Maximum latency |Average latency
            |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
            |===
            image::./histogram_total_stream_0_latency_{_subscriber_name_}.png[]
            |===
            |Number of IEC61850 Sampled Value |Minimum pacing |Maximum pacing |Average pacing
            |{_stream_} |{_minpace_} us |{_maxpace_} us |{_avgpace_} us
            |===
            """
        )

        hypervisor_lines = textwrap.dedent(
            """
            ===== Hypervisor {_hypervisor_name_}
            {{set:cellbgcolor!}}
            |===
            |Number of IEC61850 Sampled Value |Minimum latency |Maximum latency |Average latency
            |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
            |===
            image::./histogram_total_stream_0_latency_{_hypervisor_name_}.png[]
            |===
            |Number of IEC61850 Sampled Value |Minimum pacing |Maximum pacing |Average pacing
            |{_stream_} |{_minpace_} us |{_maxpace_} us |{_avgpace_} us
            |===
            """
        )

        pass_line = textwrap.dedent(
            """
            [cols="3,1",frame=all, grid=all]
            |===
            |Max latency < {_limit_} us
            |{{set:cellbgcolor:{_color_}}}{_result_}
            |{{set:cellbgcolor:transparent}}SV dropped|{_sv_dropped_}
            |===
            """
        )

        pub_sv = extract_sv(pub, streams)
        sub_sv = extract_sv(sub, streams)

        latencies, total_sv_drop = compute_latency(pub_sv, sub_sv)
        sub_pacing = compute_pacing(sub_sv)
        save_latency_histogram("latency", latencies, streams,"Total",output, sub_name)
        maxlat= compute_max(latencies[0])
        adoc_file.write(
                subcriber_lines.format(
                    _output_=output,
                    _vm_=sub_name,
                    _subscriber_name_=sub_name,
                    _stream_= get_stream_count(pub_sv),
                    _minlat_= compute_min(latencies[0]),
                    _maxlat_= maxlat,
                    _avglat_= compute_average(latencies[0]),
                    _minpace_= compute_min(sub_pacing[0]),
                    _maxpace_= compute_max(sub_pacing[0]),
                    _avgpace_= compute_average(sub_pacing[0]),
                )
        )

        if hyp is not None:
            hyp_sv = extract_sv(hyp, streams)
            hyp_latencies, total_sv_drop = compute_latency(pub_sv, hyp_sv)
            hyp_pace = compute_pacing(hyp_sv)
            adoc_file.write(
                    hypervisor_lines.format(
                        _output_=output,
                        _hypervisor_name_=hyp_name,
                        _stream_= get_stream_count(hyp_sv),
                        _minlat_= compute_min(hyp_latencies[0]),
                        _maxlat_= maxlat,
                        _avglat_= compute_average(hyp_latencies[0]),
                        _minpace_= compute_min(hyp_pace[0]),
                        _maxpace_= compute_max(hyp_pace[0]),
                        _avgpace_= compute_average(hyp_pace[0]),
                    )
            )

        if maxlat < ttot:
            adoc_file.write(
                pass_line.format(
                    _limit_=ttot,
                    _result_="PASS",
                    _color_=GREEN_COLOR,
                    _sv_dropped_=total_sv_drop
                )
            )
        else:
            adoc_file.write(
                pass_line.format(
                    _limit_=ttot,
                    _result_="FAILED",
                    _color_=RED_COLOR,
                    _sv_dropped_=total_sv_drop
                )
            )

def parse_streams(value):
    """
    Parses the `streams` argument to handle single values or ranges of values.
    Example values:
      - Single value: '0' or '3'
      - Range of values: '0..3'
    """
    if '..' in value:
        start, end = value.split('..')
        try:
            start, end = int(start), int(end)
            if start > end:
                raise argparse.ArgumentTypeError(f"Fatal: invalid stream range: {value}. Start of range must be less than end.")
            return list(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Fatal: invalid stream format: {value}. Use 'start..end' format.")
    else:
        try:
            return [int(value)]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Fatal: invalid stream stream value: {value}. Must be an integer or a range.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Latency tests report in AsciiDoc format.")
    parser.add_argument("--pub", "-p", required=True, type=str, help="SV publisher file")
    parser.add_argument("--hyp", "-y", type=str, help="SV hypervisor file")
    parser.add_argument("--hypervisor_name", type=str, help="Hypervisor name that will appear in report and graph. If not set, it will be the name of SV hypervisor file")
    parser.add_argument("--sub", "-s", type=str, help="SV subscriber file")
    parser.add_argument("--subscriber_name", type=str, help="Subscriber name that will appear in report and graph. If not set, it will be the name of SV subscriber file")
    parser.add_argument("--stream", "-S", default=[0], type=parse_streams, help="Streams to look for. If not set, only stream 0 will be considered")
    parser.add_argument("--output", "-o", default=".", type=str, help="Output directory for the generated files.")
    parser.add_argument("--ttot", default=100, type=int, help="Total latency threshold.")

    args = parser.parse_args()
    generate_adoc(args.pub, args.hyp, args.sub, args.stream, args.hypervisor_name, args.subscriber_name, args.output, args.ttot)
