import pm4py

def bpi12_filter_func(x: pm4py.objects.log.obj.Trace) -> bool:
    f1 = len(x) > 2 and len(x) <= 85

    duration = (x[-1]["time:timestamp"] - x[0]["time:timestamp"]).total_seconds()
    f2 = duration > 30 and duration <= 4052400

    f3 = x[-1]["concept:name"] not in ["A_REGISTERED", "W_Wijzigen contractgegevens"]

    f4 = "W_Wijzigen contractgegevens" not in [e["concept:name"] for e in x]
    return f1 and f2 and f3 and f4


def bpi17_filter_func(x: pm4py.objects.log.obj.Trace) -> bool:
    f1 = len(x) > 2 and len(x) <= 100

    duration = (x[-1]["time:timestamp"] - x[0]["time:timestamp"]).total_seconds()
    f2 = duration > 30 and duration <= 5137765

    f3 = x[-1]["concept:name"] not in [
        "W_Shortened completion ", "A_Denied", "O_Sent (online only)",
        "O_Sent (mail and online)", "O_Returned"
    ]

    f4 = "W_Personal Loan collection" not in [e["concept:name"] for e in x] and \
        "W_Shortened completion " not in [e["concept:name"] for e in x]
    return f1 and f2 and f3 and f4


def filter_log(log: pm4py.objects.log.obj.EventLog, log_type: str) -> pm4py.objects.log.obj.EventLog:
    if log_type == "bpi12":
        return pm4py.filter_log(bpi12_filter_func, log)
    elif log_type == "bpi17":
        return pm4py.filter_log(bpi17_filter_func, log)
    else:
        raise "Unknown log type"
