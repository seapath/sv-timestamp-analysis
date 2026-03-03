# Copyright (C) 2024, RTE (http://www.rte-france.com)
# Copyright (C) 2024 Savoir-faire Linux, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import pandas as pd

GREEN_COLOR = "#90EE90"
RED_COLOR = "#F08080"

class SvExtractor:
    def __init__(self, sv_file_path):
        self.sv_file_path = sv_file_path

    def __enter__(self):
        self.sv_file = open(self.sv_file_path, "r", encoding="utf-8")
        self._last_line = self.sv_file.readline() 
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sv_file.close()
        return False

    def extract_sv(self, streams, nb_iterations=0):
        """Extract a given number of SV timestamp data iterations from the SV data file.

        Args:
            streams (list): of stream IDs to extract
            nb_iterations (int, optional): Number of SV iterations to extract.
                Defaults to 0 which means to extract until the end of the file.

        Returns:
            A tuple containing:
            * A list of numpy arrays with extracted SV data per given stream.
              Index 0 has data for the first given stream, index 1 for the second, etc...
              For each stream, the data is as follows:
                * Index 0: numpy array of iteration number for each SV.
                * Index 1: numpy array of smpCnt for each SV.
                * Index 2: numpy array of timestamp for each SV.
            * The stream IDs found in the file. Might differ from the given streams.
        """

        if not streams:
            raise ValueError("Invalid or empty list of streams found, the -S argument might be incorrect")

        sv_content = []
        stop_parsing = False
        stop_iteration = 0

        def parse(line):
            tmp = line.rstrip().split(':')
            return (int(tmp[0]), str(tmp[1]), int(tmp[2]), int(tmp[3]))

        if self._last_line:
            sv_content.append(parse(self._last_line))
            if nb_iterations > 0:
                stop_iteration = sv_content[0][0] + nb_iterations
        else:
            # EOF was already reached, nothing should be parsed
            stop_parsing = True

        while not stop_parsing:
            line = self.sv_file.readline()
            if line:
                sv = parse(line)
                if stop_iteration == 0 or sv[0] < stop_iteration:
                    sv_content.append(sv)
                    continue

            # EOF or stop_iteration reached
            self._last_line = line
            stop_parsing = True

        sv_it = np.array([i[0] for i in sv_content])
        sv_id = np.array([i[1] for i in sv_content])
        sv_cnt = np.array([i[2] for i in sv_content])
        sv_timestamps = np.array([i[3] for i in sv_content])

        stream_names = np.unique(sv_id)
        sv = []
        for s in streams:
            ids_occur = np.where(sv_id == f"{s:04x}")
            sv.append([sv_it[ids_occur], sv_cnt[ids_occur], sv_timestamps[ids_occur]])

        return sv, stream_names


def verify_sv_logs_consistency(sv_filename_1, sv_filename_2):
    """Verify that both SV files are comparables. It means that they contain the same number of iterations.

    If they do not have the same number of iterations, it can mean:
    * Packets were re-ordered.
    * Too many SV were lost.
    In that case, latencies cannot be computed, because some received SV cannot
    be linked correctly to a published SV.

    Raises:
        ValueError: If the two SV files do not have the same number of iterations.
        CalledProcessError: If the "tail" command failed.

    Returns:
        None
    """

    tail_1 = subprocess.run(
        ["tail", "-n", "1", sv_filename_1],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tail_2 = subprocess.run(
        ["tail", "-n", "1", sv_filename_2],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    last_it_1 = tail_1.stdout.rstrip().split(":")[0]
    last_it_2 = tail_2.stdout.rstrip().split(":")[0]

    if not last_it_1:
        raise ValueError(f"{sv_filename_1} has no valid data.")
    if not last_it_2:
        raise ValueError(f"{sv_filename_2} has no valid data.")

    if last_it_1 != last_it_2:
        raise ValueError(
            f"{sv_filename_1} has {last_it_1} iterations but {sv_filename_2} has {last_it_2}"
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

def save_latency_histogram(df, stream, sub_name, output, threshold=0):
    """Save a latency histogram for an SV stream from a latency dataframe.

    Args:
        df (DataFrame): Latency dataframe.
        stream (int): SV stream ID.
        sub_name (str): Subscriber name.
        output (str): Output path where the histogram will be saved in a "results" subdirectory.
        threshold (int, optional): Threshold value that will be represented as a vertical red line.
            If <= 0, this threshold isn't showed. Defaults to 0.

    Returns:
        str: Full file path of the save histogram.
    """

    plt.hist(df["latency"], bins=20, weights=df["count"], alpha=0.7)

    plt.xlabel("Latency (us)")
    plt.ylabel("Occurrences")
    plt.yscale("log")
    plt.title(f"{sub_name} SV stream 0x{stream:04x} latency histogram")

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

        pass_line_max_latency = textwrap.dedent(
            """
            [cols="3,1",frame=all, grid=all]
            |===
            |Max latency < {_limit_} us
            |{{set:cellbgcolor:{_color_}}}{_result_}
            |{{set:cellbgcolor:transparent}}SV dropped|{_sv_dropped_}
            |===
            """
        )

        pass_line_min_latency = textwrap.dedent(
            """
            [cols="3,1",frame=all, grid=all]
            |===
            |Min latency > 0 us
            |{{set:cellbgcolor:{_color_}}}{_result_}
            |===
            """
        )

        verify_sv_logs_consistency(pub, sub)

        latencies_df = [ pd.DataFrame({"latency": [], "count": []}) for _ in range(len(streams)) ]
        total_sv_drop = 0

        with SvExtractor(pub) as pub_extractor, SvExtractor(sub) as sub_extractor:
            chunk_size = 100

            pub_sv, _ = pub_extractor.extract_sv(streams, chunk_size)
            sub_sv, sub_stream_names = sub_extractor.extract_sv(streams, chunk_size)

            while len(sub_stream_names) > 0:
                chunk_latencies, sv_drop = compute_latency(pub_sv, sub_sv)

                total_sv_drop += sv_drop

                for i in range(len(streams)):
                    val, counts = np.unique(chunk_latencies[i], return_counts=True)
                    df = pd.DataFrame({"latency": val, "count": counts})

                    # Merge chunk latencies with global counts
                    latencies_df[i] = pd.merge(latencies_df[i], df, on="latency", how="outer", suffixes=("", "_chunk"))
                    latencies_df[i] = latencies_df[i].fillna(0) # Outer merge introduces NaN when a latency doesn't exist in one of the two tables
                    latencies_df[i]["count"] += latencies_df[i].pop("count_chunk")

                    # Introduction of NaN values in merge changed data type to float64
                    latencies_df[i] = latencies_df[i].astype(np.int64)

                # Next chunk
                pub_sv, _ = pub_extractor.extract_sv(streams, chunk_size)
                sub_sv, sub_stream_names = sub_extractor.extract_sv(streams, chunk_size)

        for i in range(len(streams)):
            if display_threshold:
                save_latency_histogram(latencies_df[i], streams[i], sub_name, output, max_latency_threshold)
            else:
                save_latency_histogram(latencies_df[i], streams[i], sub_name, output)

        maxlat= compute_max(latencies[0])
        minlat = compute_min(latencies[0])
        adoc_file.write(
                subcriber_lines.format(
                    _output_=output,
                    _subscriber_name_=sub_name,
                    _stream_id_= sub_stream_names[0],
                    _stream_ = streams[0],
                    _minlat_= minlat,
                    _maxlat_= maxlat,
                    _avglat_= compute_average(latencies[0]),
                    _minpace_= compute_min(sub_pacing[0]),
                    _maxpace_= compute_max(sub_pacing[0]),
                    _avgpace_= compute_average(sub_pacing[0]),
                )
        )

        if hyp is not None:
            verify_sv_logs_consistency(pub, hyp)

            hyp_sv = []
            hyp_stream_names = []
            with SvExtractor(hyp) as hyp_extractor:
                hyp_sv, hyp_stream_names = hyp_extractor.extract_sv(streams)

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
                pass_line_max_latency.format(
                    _limit_=max_latency_threshold,
                    _result_="PASS",
                    _color_=GREEN_COLOR,
                    _sv_dropped_=total_sv_drop
                )
            )
        else:
            adoc_file.write(
                pass_line_max_latency.format(
                    _limit_=max_latency_threshold,
                    _result_="FAILED",
                    _color_=RED_COLOR,
                    _sv_dropped_=total_sv_drop
                )
            )

        if minlat > 0:
            adoc_file.write(
                pass_line_min_latency.format(
                    _result_="PASS",
                    _color_=GREEN_COLOR,
                )
            )
        else:
            adoc_file.write(
                pass_line_min_latency.format(
                    _result_="FAILED",
                    _color_=RED_COLOR,
                )
            )

def parse_streams(value):
    """
    Parses the `streams` argument to handle single values or ranges of values.
    Example values:
      - Single value: '0', '3' or '40ff'
      - Range of values: '0..3' or '4000..40ff'
    """
    if ".." in value:
        start, end = value.split("..")
        try:
            start, end = int(start, 16), int(end, 16)
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
            return [int(value, 16)]
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
        help="Streams (SVID) to consider as hexadecimal (e.g. '0', '3', '4000..40ff'). If not set, only stream 0 will be considered",
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
