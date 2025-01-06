# Copyright (C) 2024, RTE (http://www.rte-france.com)
# Copyright (C) 2024 Savoir-faire Linux, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import pandas as pd

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

    return sv, stream_names

def verify_sv_logs_consistency(sv_data_1, sv_data_2, sv_filename_1, sv_filename_2):
# Verify that both sv files are comparables. It means:
# - contains the same number of streams
# - contains the same number of iterations
# If they do not have the same number of iterations, it can mean :
# - packets reordering
# - too many SV lost.
# In both cases, the latency cannot be computed, because a received SV cannot
# be linked correctly to a published SV.

    # Check for same number of streams
    if len(sv_data_1) != len(sv_data_2):
        raise ValueError(
            f"{sv_filename_1} has {len(sv_data_1)} stream, but {sv_filename_2} has {len(sv_data_2)}'"
        )

    # Check last iteration counter
    for stream in range(0, len(sv_data_1)):
        # Compare last value of the iteration columns
        if sv_data_1[stream][0][-1] != sv_data_2[stream][0][-1]:
            raise ValueError(
                f"{sv_filename_1} and {sv_filename_2} don't have the same number of iterations"
            )


def handle_sv_drop(pub_stream, sub_stream):
    # Compute the latency on a stream with sv lost
    # All the magic remains in the pandas dataframe merge function using the
    # inner method to combine tables. This function handles the missalignement
    # between subscriber and publisher values.
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

    columns = ["iteration", "counter", "time"]
    pub_data = pd.DataFrame(pub_stream, index=columns).T
    sub_data = pd.DataFrame(sub_stream, index=columns).T

    merged_data = pd.merge(pub_data, sub_data, on=["iteration", "counter"], how="inner")
    latencies = merged_data["time_y"] - merged_data["time_x"]

    return np.array(latencies)


def compute_latency(pub_sv, sub_sv):

    latencies = [[]] * len(pub_sv)
    sv_drop = 0

    for stream in range(0, len(pub_sv)):

        pub_sv_stream = pub_sv[stream]
        sub_sv_stream = sub_sv[stream]
        sv_drop_stream = abs(len(pub_sv_stream[0]) - len(sub_sv_stream[0]))
        sv_drop += sv_drop_stream

        if sv_drop_stream > 0:
            # if sv drop is detected on this stream, pandas will be used to
            # reconstruct the link between data and compute the latency
            # It will take additionnal times to convert from numpy to pandas,
            # so this is only done when there is sv drop.
            latencies[stream] = handle_sv_drop(pub_sv_stream, sub_sv_stream)
        else:
            latencies[stream] = sub_sv_stream[2] - pub_sv_stream[2]

    return latencies, sv_drop

def compute_pacing(sv):
    streams = len(sv)
    pacing = [[0]] * len(sv)
    for stream in range(0, streams):
        pacing[stream] = np.diff(sv[stream][2])
    return pacing

def compute_min(values):
    return np.min(values) if values.size > 0 else None

def compute_max(values):
    return np.max(values) if values.size > 0 else None

def compute_average(values):
    return np.round(np.mean(values)) if values.size > 0 else None

def compute_neglat(values):
    return np.count_nonzero(values < 0)

def save_latency_histogram(values, streams, stream_names, sub_name, output, threshold=0):

    for stream, value in zip(streams, values):
        plt.hist(value, bins=20, alpha=0.7)

        plt.xlabel(f"Latency (us)")
        plt.ylabel("Occurrences")
        plt.yscale('log')
        plt.title(f"{sub_name} SV stream {stream_names[stream]} latency histogram")

        if threshold > 0:
            plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Limit ({threshold} us)')
            plt.legend()

        filename = f"histogram_{sub_name}_stream_{stream}_latency.png"
        filepath = os.path.realpath(f"{output}/results/{filename}")
        plt.savefig(filepath)
        print(f"Histogram saved as {filename}.")
        plt.close()

    return filepath

def generate_adoc(pub, hyp, sub, streams, hyp_name, sub_name, output, max_latency_threshold, display_threshold):
    if not os.path.exists(f'{output}/results'):
        os.makedirs(f'{output}/results')

    with open(f"{output}/results/latency_tests.adoc", "w", encoding="utf-8") as adoc_file:
        subcriber_lines = textwrap.dedent(
            """
            ===== Subscriber {_subscriber_name_}
            {{set:cellbgcolor!}}
            |===
            |IEC61850 Sampled Value Stream |Minimum latency |Maximum latency |Average latency
            |{_stream_id_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
            |===
            image::./histogram_{_subscriber_name_}_stream_{_stream_}_latency.png[]
            |===
            |IEC61850 Sampled Value Stream |Minimum pacing |Maximum pacing |Average pacing
            |{_stream_id_} |{_minpace_} us |{_maxpace_} us |{_avgpace_} us
            |===
            """
        )

        hypervisor_lines = textwrap.dedent(
            """
            ===== Hypervisor {_hypervisor_name_}
            {{set:cellbgcolor!}}
            |===
            |IEC61850 Sampled Value Stream |Minimum latency |Maximum latency |Average latency
            |{_stream_id_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
            |===
            image::./histogram_{_hypervisor_name_}_stream_{_stream_}_latency.png[]
            |===
            |IEC61850 Sampled Value Stream |Minimum pacing |Maximum pacing |Average pacing
            |{_stream_id_} |{_minpace_} us |{_maxpace_} us |{_avgpace_} us
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

        pub_sv, _ = extract_sv(pub, streams)
        sub_sv, sub_stream_names = extract_sv(sub, streams)
        verify_sv_logs_consistency(pub_sv, sub_sv, pub, sub)

        latencies, total_sv_drop = compute_latency(pub_sv, sub_sv)
        sub_pacing = compute_pacing(sub_sv)
        if display_threshold:
            save_latency_histogram(latencies, streams, sub_stream_names, sub_name, output, max_latency_threshold)
        else:
            save_latency_histogram(latencies, streams, sub_stream_names, sub_name, output)
        maxlat= compute_max(latencies[0])
        adoc_file.write(
                subcriber_lines.format(
                    _output_=output,
                    _subscriber_name_=sub_name,
                    _stream_id_= sub_stream_names[0],
                    _stream_ = streams[0],
                    _minlat_= compute_min(latencies[0]),
                    _maxlat_= maxlat,
                    _avglat_= compute_average(latencies[0]),
                    _minpace_= compute_min(sub_pacing[0]),
                    _maxpace_= compute_max(sub_pacing[0]),
                    _avgpace_= compute_average(sub_pacing[0]),
                )
        )

        if hyp is not None:
            hyp_sv, hyp_stream_names = extract_sv(hyp, streams)
            verify_sv_logs_consistency(pub_sv, hyp_sv, pub, hyp)
            hyp_latencies, total_sv_drop = compute_latency(pub_sv, hyp_sv)
            hyp_pace = compute_pacing(hyp_sv)
            adoc_file.write(
                    hypervisor_lines.format(
                        _output_=output,
                        _hypervisor_name_=hyp_name,
                        _stream_id_= hyp_stream_names[0],
                        _stream_ = streams[0],
                        _minlat_= compute_min(hyp_latencies[0]),
                        _maxlat_= maxlat,
                        _avglat_= compute_average(hyp_latencies[0]),
                        _minpace_= compute_min(hyp_pace[0]),
                        _maxpace_= compute_max(hyp_pace[0]),
                        _avgpace_= compute_average(hyp_pace[0]),
                    )
            )

        if maxlat < max_latency_threshold:
            adoc_file.write(
                pass_line.format(
                    _limit_=max_latency_threshold,
                    _result_="PASS",
                    _color_=GREEN_COLOR,
                    _sv_dropped_=total_sv_drop
                )
            )
        else:
            adoc_file.write(
                pass_line.format(
                    _limit_=max_latency_threshold,
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
    if ".." in value:
        start, end = value.split("..")
        try:
            start, end = int(start), int(end)
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Fatal: invalid stream range: {value}. Start of range must be less than end."
                )
            return list(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Fatal: invalid stream format: {value}. Use 'start..end' format."
            )
    else:
        try:
            return [int(value)]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Fatal: invalid stream stream value: {value}. Must be an integer or a range."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute latencies from sv_timestamp_logger and generate latency tests report in AsciiDoc format"
    )
    parser.add_argument(
        "--pub", "-p", required=True, type=str, help="SV publisher file"
    )
    parser.add_argument("--hyp", "-y", type=str, help="SV hypervisor file")
    parser.add_argument("--sub", "-s", type=str, help="SV subscriber file")
    parser.add_argument(
        "--hypervisor_name",
        type=str,
        help="Hypervisor name that will appear in report and graph. If not set, it will be the name of SV hypervisor file",
    )
    parser.add_argument(
        "--subscriber_name",
        type=str,
        help="Subscriber name that will appear in report and graph. If not set, it will be the name of SV subscriber file",
    )
    parser.add_argument(
        "--stream",
        "-S",
        default=[0],
        type=parse_streams,
        help="Streams to consider. If not set, only stream 0 will be considered",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".",
        type=str,
        help="Output directory for the generated files.",
    )
    parser.add_argument(
        "--max_latency", "-m", default=100, type=int, help="Maximum latency threshold"
    )
    parser.add_argument(
        "--display_max_latency",
        action="store_true",
        help="Display max latency threshold on histograms if set"
    )

    args = parser.parse_args()
    if not args.hypervisor_name:
        hyp_name=args.hyp
    else:
        hyp_name=args.hypervisor_name
    if not args.subscriber_name:
        sub_name=args.sub
    else:
        sub_name=args.subscriber_name

    generate_adoc(
        args.pub,
        args.hyp,
        args.sub,
        args.stream,
        hyp_name,
        sub_name,
        args.output,
        args.max_latency,
        args.display_max_latency,
    )
