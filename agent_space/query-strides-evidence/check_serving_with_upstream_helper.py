"""Use the existing upstream model comparison helper without overriding defaults."""
from pathlib import Path
import argparse
import json
from tests.models.utils import check_logprobs_close


def main():
    parser=argparse.ArgumentParser();parser.add_argument('candidate',type=Path);parser.add_argument('baseline',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    results=[]
    for path in [args.candidate,args.baseline]:
        rows=sorted(json.loads(path.read_text())['choices'],key=lambda item:item['index'])
        results.append([(row['logprobs']['tokens'],row['text'],row['logprobs']['top_logprobs']) for row in rows])
    check_logprobs_close(outputs_0_lst=results[0],outputs_1_lst=results[1],name_0='candidate',name_1='baseline')
    args.output.write_text(json.dumps({'upstream_helper':'tests.models.utils.check_logprobs_close',
        'defaults_unchanged':True,'passed':True,
        'exact_texts':all(a[1]==b[1] for a,b in zip(*results)),
        'limits':'Uses the unchanged upstream common-prefix/top-logprob criterion; not an assertion of bitwise model equivalence.'},indent=2)+'\n')


if __name__=='__main__':main()
