from torch import LongTensor
import editdistance

def cer_score(target_seq: LongTensor, predicted_seq: LongTensor) -> float:
    if len(target_seq) == 0 and len(predicted_seq) == 0:
        return 1.0
    return editdistance.eval(target_seq, predicted_seq) / len(target_seq)
