"""Pinned downloads and story-level split; no validation text used for fitting."""
import shutil
import urllib.request
import sentencepiece as spm
from common import *

SOURCES={
 'model.safetensors':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors',
 'tokenizer.json':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json',
 'tok512.model':'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model',
 'TinyStoriesV2-GPT4-valid.txt':'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt',
 'chinchilla-runs.csv':'https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv'}

def tokenizer():return spm.SentencePieceProcessor(model_file=str(DATA/'tok512.model'))

def main():
    provenance=[]
    for name,url in SOURCES.items():
        path=DATA/name
        if not path.exists():
            print('Downloading',name,flush=True)
            temporary=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:shutil.copyfileobj(response,f)
            temporary.replace(path)
        provenance.append({'file':str(path.relative_to(ROOT)),'url':url,'bytes':path.stat().st_size,'sha256':sha256(path)})
    stories=[story.strip() for story in (DATA/'TinyStoriesV2-GPT4-valid.txt').read_text().split('<|endoftext|>') if story.strip()]
    split=int(.9*len(stories));sp=tokenizer();assert sp.vocab_size()==512
    report={'stories':len(stories),'training_stories':split,'validation_stories':len(stories)-split,
            'split_rule':'first 90% of nonempty stripped whole stories, in file order',
            'vocabulary':sp.vocab_size(),'eos_id':sp.eos_id(),'bos_id':sp.bos_id(),
            'special_ids':[sp.unk_id(),sp.bos_id(),sp.eos_id()],'files':provenance,'environment':environment()}
    mismatches=[]
    for name,texts in [('train',stories[:split]),('validation',stories[split:])]:
        encoded=sp.encode(texts,out_type=int,num_threads=8)
        lengths=np.array([len(tokens)+1 for tokens in encoded],dtype=np.int64)
        stream=np.fromiter((v for tokens in encoded for v in (*tokens,sp.eos_id())),dtype=np.int32,count=int(lengths.sum()))
        np.save(ART/f'{name}-tokens.npy',stream)
        np.save(ART/f'{name}-story-ends.npy',np.cumsum(lengths))
        chars=sum(map(len,texts))
        report[name]={'tokens':len(stream),'characters':chars,'characters_per_token':chars/len(stream),
                      'sha256_tokens':sha256(ART/f'{name}-tokens.npy')}
        if name=='validation':
            for index,(text,tokens) in enumerate(zip(texts,encoded)):
                decoded=sp.decode(tokens)
                if decoded!=text:mismatches.append({'index':index,'original':text,'decoded':decoded})
    report['validation_decode_mismatches']=len(mismatches)
    write_json(ART/'tokenizer-mismatches.json',mismatches)
    write_json(EVIDENCE/'data-provenance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','environment']},indent=2),flush=True)
    assert (split,len(stories)-split)==(24867,2763)
    assert len(mismatches)==3

if __name__=='__main__':main()
