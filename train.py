"""Run: python train.py. All evaluation is reproducible and saved in outputs/."""
from pathlib import Path
import hashlib,json,time,urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib,sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (precision_score,recall_score,f1_score,accuracy_score,
                            average_precision_score,confusion_matrix,precision_recall_curve)
from sklearn.dummy import DummyClassifier
from core import normalized_text,template_key,make_model,warning_cues,explain

ROOT=Path(__file__).resolve().parent
SOURCE='https://raw.githubusercontent.com/AbeerAlhuthali/PhishNet/main/PhishNet_Dataset1_original.csv'

def metrics(y,scores,threshold):
    pred=(np.asarray(scores)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                average_precision=average_precision_score(y,scores),false_positive_rate=fp/max(tn+fp,1),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def savefig(name):
    plt.tight_layout();plt.savefig(ROOT/'outputs'/f'{name}.png',dpi=250,bbox_inches='tight');plt.close()

def run():
    for folder in ['data','outputs','models']: (ROOT/folder).mkdir(exist_ok=True)
    rawpath=ROOT/'data/raw.csv'
    if not rawpath.exists():
        with urllib.request.urlopen(SOURCE,timeout=30) as response: rawpath.write_bytes(response.read())
    data=pd.read_csv(rawpath)
    data['original_label']=data.LABEL.astype(str).str.lower().str.strip()
    if not set(data.original_label)<= {'ham','spam','smishing'}: raise ValueError('Unexpected label')
    data['target']=(data.original_label=='smishing').astype(int)
    data['normalized']=data.TEXT.map(normalized_text)
    initial=len(data)
    empty=data.TEXT.isna() | data.normalized.eq('')
    data=data.loc[~empty].copy()
    conflicts=data.groupby('normalized').target.nunique()
    keys=conflicts[conflicts>1].index
    ambiguous=data[data.normalized.isin(keys)]
    ambiguous.to_csv(ROOT/'outputs/ambiguous_excluded.csv',index=False)
    data=data[~data.normalized.isin(keys)].copy()
    before=len(data); data=data.drop_duplicates('normalized').copy()
    data['group']=data.TEXT.map(template_key)
    audit={'original_rows':initial,'raw_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),
           'mirror_url':SOURCE,'missing_or_empty_rows':int(empty.sum()),
           'conflicting_binary_text_groups':len(keys),'ambiguous_rows_excluded':len(ambiguous),
           'duplicate_text_rows_removed':before-len(data),'retained_rows':len(data),
           'original_class_counts':pd.read_csv(rawpath).LABEL.str.lower().value_counts().to_dict(),
           'retained_binary_counts':data.target.value_counts().to_dict()}
    data=data.reset_index(drop=True)
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    dev_idx,test_idx=next(outer.split(data.TEXT,data.target,data.group))
    dev=data.iloc[dev_idx]
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=43)
    tr_rel,va_rel=next(inner.split(dev.TEXT,dev.target,dev.group))
    train=dev.iloc[tr_rel];val=dev.iloc[va_rel];test=data.iloc[test_idx]
    for a,b in [(train,val),(train,test),(val,test)]:
        assert set(a.group).isdisjoint(b.group),'Template leakage'
    for name,part in [('train',train),('validation',val),('test',test)]:
        part.to_csv(ROOT/'data'/f'{name}.csv',index=False)
    audit['split_counts']={n: {'rows':len(p),'phishing':int(p.target.sum())} for n,p in [('train',train),('validation',val),('test',test)]}
    (ROOT/'outputs/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    modelspecs=[('Word NB','word','nb'),('Word Logistic Regression','word','lr'),
                ('Word SVM','word','svm'),('Character Logistic Regression','char','lr'),
                ('Word + Character LR','combined','lr'),('Hybrid Logistic Regression','hybrid','lr')]
    # Select representations and thresholds using validation only. Test is used afterwards.
    trained={};selection=[]
    for name,kind,clf in modelspecs:
        start=time.perf_counter();model=make_model(kind,clf);model.fit(train.TEXT,train.target)
        elapsed=time.perf_counter()-start
        if clf=='svm':
            scores=model.decision_function(val.TEXT);candidates=np.linspace(-2,2,81)
        else:
            scores=model.predict_proba(val.TEXT)[:,1];candidates=np.linspace(.05,.95,91)
        evaluations=[(float(t),metrics(val.target,scores,t)) for t in candidates]
        threshold,perf=max(evaluations,key=lambda item:(item[1]['f1'],item[1]['precision']))
        selection.append({'model':name,'threshold':threshold,'validation_f1':perf['f1'],
                          'validation_precision':perf['precision'],'validation_recall':perf['recall'],'fit_seconds':elapsed})
        trained[name]=(model,threshold,clf)
    selection=pd.DataFrame(selection);selection.to_csv(ROOT/'outputs/validation_selection.csv',index=False)
    # Deployment is restricted to interpretable LR models; select by validation F1.
    eligible=selection[selection.model.str.contains('Logistic Regression|LR',regex=True)]
    chosen=eligible.sort_values(['validation_f1','validation_precision'],ascending=False).iloc[0]['model']
    results=[];saved_scores={}
    for row in selection.to_dict('records'):
        name=row['model'];model,threshold,clf=trained[name]
        scores=model.decision_function(test.TEXT) if clf=='svm' else model.predict_proba(test.TEXT)[:,1]
        results.append({'model':name,'threshold':threshold,**metrics(test.target,scores,threshold),'fit_seconds':row['fit_seconds']})
        saved_scores[name]=scores
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),train.target)
    scores=dummy.predict_proba(np.zeros((len(test),1)))[:,1]
    results.append({'model':'Majority baseline','threshold':.5,**metrics(test.target,scores,.5),'fit_seconds':0})
    rulescores=np.asarray([float(len(warning_cues(x))>=2) for x in test.TEXT])
    results.append({'model':'Two-cue rule baseline','threshold':.5,**metrics(test.target,rulescores,.5),'fit_seconds':0})
    results=pd.DataFrame(results);results.to_csv(ROOT/'outputs/test_results.csv',index=False)
    model,threshold,_=trained[chosen]
    joblib.dump({'model':model,'threshold':threshold,'name':chosen,'sklearn_version':sklearn.__version__},ROOT/'models/scamshield.joblib')
    predictions=test[['TEXT','original_label','target']].copy()
    predictions['score']=saved_scores[chosen];predictions['prediction']=(predictions.score>=threshold).astype(int)
    predictions.to_csv(ROOT/'outputs/test_predictions.csv',index=False)
    subtype=[]
    for label,part in predictions.groupby('original_label'):
        subtype.append({'original_label':label,'rows':len(part),'warning_rate':float(part.prediction.mean())})
    pd.DataFrame(subtype).to_csv(ROOT/'outputs/subtype_warning_rates.csv',index=False)
    (ROOT/'outputs/model_metadata.json').write_text(json.dumps({'selected_model':chosen,'threshold':threshold,
      'selection_policy':'Highest validation F1 among logistic regression representations; precision breaks ties',
      'warning_threshold_policy':'Validation F1 optimisation; no claim of calibrated real-world scam probability',
      'seed':42,'test_metrics':results[results.model==chosen].iloc[0].to_dict()},indent=2),encoding='utf-8')
    # All figures use measured data.
    plt.figure(figsize=(6,3.5));data.original_label.value_counts().plot.bar(color=['#245b85','#b54b35','#c18a26'])
    plt.ylabel('Retained messages');plt.xlabel('Original class');plt.xticks(rotation=0);savefig('class_distribution')
    plt.figure(figsize=(6,3.5))
    for target,label in [(0,'Non-phishing'),(1,'Phishing')]:
        plt.hist(data[data.target==target].TEXT.str.len(),bins=35,alpha=.55,density=True,label=label)
    plt.xlabel('Message length (characters)');plt.ylabel('Density');plt.legend();savefig('message_lengths')
    cm=confusion_matrix(test.target,predictions.prediction,labels=[0,1])
    plt.figure(figsize=(5,4));plt.imshow(cm,cmap='Blues');plt.colorbar()
    for i in range(2):
        for j in range(2):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=15)
    plt.xticks([0,1],['Non-phishing','Phishing']);plt.yticks([0,1],['Non-phishing','Phishing'])
    plt.xlabel('Predicted');plt.ylabel('Actual');plt.title(chosen);savefig('confusion_matrix')
    plt.figure(figsize=(6,4))
    for name in ['Word Logistic Regression',chosen]:
        if name=='Word Logistic Regression' and chosen==name:continue
        precision,recall,_=precision_recall_curve(test.target,saved_scores[name])
        plt.plot(recall,precision,label=name)
    plt.axhline(test.target.mean(),linestyle='--',color='gray',label='Test phishing prevalence')
    plt.xlabel('Recall');plt.ylabel('Precision');plt.legend(fontsize=8);savefig('precision_recall')
    plt.figure(figsize=(8,4));plt.barh(results.model,results.f1,color='#245b85');plt.xlabel('Test phishing F1');plt.xlim(0,1);savefig('model_comparison')
    # Feature contributions are exact additive contributions to the LR logit.
    example=predictions[(predictions.target==1)&(predictions.prediction==1)].iloc[0].TEXT if ((predictions.target==1)&(predictions.prediction==1)).any() else predictions.iloc[0].TEXT
    explanation=explain(model,example)
    pd.DataFrame(explanation).to_csv(ROOT/'outputs/example_explanation.csv',index=False)
    plt.figure(figsize=(7,3.5));plt.barh([x['feature'] for x in explanation][::-1],[x['contribution'] for x in explanation][::-1],color='#b54b35')
    plt.xlabel('Contribution towards phishing logit');savefig('feature_contributions')
    print(json.dumps(audit,indent=2));print(results[['model','precision','recall','f1','false_positive_rate']].round(4).to_string(index=False));print('Selected:',chosen)
    return audit,results

if __name__=='__main__':run()
