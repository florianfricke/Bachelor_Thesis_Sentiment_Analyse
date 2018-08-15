Bitte im RAW Modus anschauen!

## emphrasis and stopwords
### Lexicon-Method
confusion matrix:
[[271 182  48]
 [799 725 333]
 [314 282 207]]

classification report:
             precision    recall  f1-score   support

   negative       0.35      0.26      0.30       803
    neutral       0.61      0.39      0.48      1857
   positive       0.20      0.54      0.29       501

avg / total       0.48      0.38      0.40      3161

### textblob-de
confusion matrix:
[[228 243  30]
 [552 998 307]
 [241 368 194]]

classification report:
             precision    recall  f1-score   support

   negative       0.37      0.24      0.29       803
    neutral       0.62      0.54      0.58      1857
   positive       0.22      0.46      0.30       501

avg / total       0.49      0.45      0.46      3161

### Multinomial Naive Bayes
model accuracy: 0.6406

confusion matrix:
[[ 296  158   47]
 [ 149 1317  391]
 [  71  320  412]]

classification report:
             precision    recall  f1-score   support

   negative       0.48      0.51      0.50       803
    neutral       0.73      0.71      0.72      1857
   positive       0.57      0.59      0.58       501

avg / total       0.65      0.64      0.64      3161
