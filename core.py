"""Shared ScamShield features and explanations. Used by training and the demo."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

FEATURE_NAMES = ['url_present', 'credential_request', 'urgency', 'reward_offer',
                 'advance_payment', 'account_threat', 'length', 'digit_fraction']
PATTERNS = {
 'url_present': r'https?://|www\.|hxxps?://',
 'credential_request': r'\b(?:share|send|provide|disclose|enter)\b.{0,35}\b(?:otp|pin|password|cvv|credentials)\b',
 'urgency': r'\b(?:urgent|immediately|expire|expires|last chance|act now)\b',
 'reward_offer': r'\b(?:winner|won|lottery|prize|reward|cashback)\b',
 'advance_payment': r'\b(?:processing fee|registration fee|advance fee|pay.{0,20}(?:claim|release))\b',
 'account_threat': r'\b(?:account|card)\b.{0,35}\b(?:block|blocked|suspend|suspended|deactivate|deactivated)\b',
}

def normalized_text(text):
    return re.sub(r'\s+', ' ', str(text).lower()).strip()

def template_key(text):
    """Conservative deterministic grouping: collapse links and digit sequences."""
    text = normalized_text(text)
    text = re.sub(r'(?:https?://|www\.)\S+', '<url>', text)
    return re.sub(r'\d+', '<num>', text)

class StructuralFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows=[]
        for value in X:
            text=str(value); low=text.lower()
            row=[float(bool(re.search(PATTERNS[name], low))) for name in FEATURE_NAMES[:6]]
            row += [min(len(text),1000)/1000, sum(c.isdigit() for c in text)/max(len(text),1)]
            rows.append(row)
        return np.asarray(rows, dtype=float)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES,dtype=object)

def representation(kind):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=15000,sublinear_tf=True)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=20000,sublinear_tf=True)
    if kind=='word': return word
    if kind=='char': return char
    components=[('word',word),('char',char)]
    if kind=='hybrid': components.append(('structure',StructuralFeatures()))
    return FeatureUnion(components)

def make_model(kind='hybrid', classifier='lr'):
    choices={
      'lr':LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=42),
      'svm':LinearSVC(C=1.0,class_weight='balanced',random_state=42),
      'nb':MultinomialNB(alpha=1.0)}
    return Pipeline([('features',representation(kind)),('classifier',choices[classifier])])

def warning_cues(text):
    labels={'url_present':'Contains a link', 'credential_request':'Language requesting credentials',
      'urgency':'Urgency language','reward_offer':'Reward or prize language',
      'advance_payment':'Advance-payment language','account_threat':'Account restriction threat'}
    return [labels[k] for k,p in PATTERNS.items() if re.search(p,str(text).lower())]

def explain(model,text,limit=8):
    features=model.named_steps['features']
    names=features.get_feature_names_out()
    row=features.transform([text])
    contributions=row.multiply(model.named_steps['classifier'].coef_[0]).toarray().ravel()
    indices=np.argsort(contributions)[::-1]
    return [{'feature':str(names[i]),'contribution':float(contributions[i])}
            for i in indices if contributions[i]>0][:limit]
