<!--Copyright (C) 2024 Savoir-faire Linux, Inc.
SPDX-License-Identifier: Apache-2.0 -->

# sv-timetamp-analysis

sv-timetamp-analysis is a tool used to analyze IEC61850 Sample Values
timestamps recorded by the sv_timestamp_logger
(https://github.com/seapath/sv_timestamp_logger) tool and generate a
report summarizing the latencies of the SVs recorded.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Release notes](#release-notes)

## Introduction
## Features

- Compute various latency metrics on IEC61850 SV (pacing, latency
  histogram, statistics)
- Detection and correction of dropped SV
- Support of high range of SV computation

## Installation
### Requirements

Following Python packages are needed:
```bash
pip install \
  numpy \
  matplotlib
```

## Usage

First, generate IEC61850 SV data timestamps using sv_timestamp_logger
tool. At least Publisher SV timestamps and subscriber SV timestamps are
needed to generate a latency report. Optionally, hypervisor SV
timestamps can also be used to get a measurement of latency between
publisher and subscriber machines.

Then, you can run sv-timetamp-analysis tool by running:
```bash
python3 sv_timestamp_analysis.py \
 --pub ts_sv_publisher.txt \
 --sub ts_sv_subscriber_guest0.txt
```

After computation, results are available by default in current
directory in `results` folder. You can override this setting using `-o`
argument.
A report is generated in .adoc format containing .png histogram of the SV
streams computed. On each stream, a latency threshold test is computed
based on the value of the `--ttot` argument (by default, 100µs).

By default, only stream 0 is used to compute latencies. You can
override this setting using `-S` or `--stream` argument.

## Release notes
### Version v0.1
Initial release
