from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import numpy as np

def choquet(solution, pred1, pred2, pred3, labels):
    solution = solution.reshape((-1))
    fuzzymeasures = np.array([solution[0],solution[1],solution[2]])
    #fuzzymeasures = np.array([0.09,0.08,0.07])
    l = Symbol('l', real = True)
    lam = solve(( 1 + l* fuzzymeasures[0]) * ( 1 + l* fuzzymeasures[1]) *( 1 + l* fuzzymeasures[2]) - (l+1), l)
    if len(lam) < 3:
      lam = np.asarray(lam)
    else:
      if lam[0] == 0:
          lam = lam = np.asarray(lam[2])
      elif lam[1] == 0:
          lam = np.asarray(lam[2])
      elif lam[2] == 0:
          lam = np.asarray(lam[1])

    Ypred_fuzzy = np.zeros(shape = pred1.shape, dtype = float)
    for sample in range(0,pred1.shape[0]):
        for classes in range(0,2):
            scores = np.array([pred1[sample][classes], pred2[sample][classes], pred3[sample][classes]])
            permutedidx = np.flip(np.argsort(scores))
            scoreslambda = scores[permutedidx]
            fmlambda = fuzzymeasures[permutedidx]
            ge_prev = fmlambda[0]
            fuzzyprediction = scoreslambda[0] * fmlambda[0]
            for i in range(1,2):
                ge_curr = ge_prev + fmlambda[i] + lam * fmlambda[i] * ge_prev
                fuzzyprediction = fuzzyprediction + scoreslambda[i] *(ge_curr - ge_prev)
                ge_prev = ge_curr
            fuzzyprediction = fuzzyprediction + scoreslambda[2] * ( 1 - ge_prev)
            Ypred_fuzzy[sample][classes] = fuzzyprediction
    ypred_fuzzy = np.argmax(Ypred_fuzzy, axis=1)
    pred_label = []
    for i in ypred_fuzzy:
      label = np.zeros((2))
      label[i] = label[i]+1
      pred_label.append(label)
    pred_label = np.array(pred_label)
    acc = accuracy_score(labels,pred_label)
    #print(acc)
    return -acc

def sugeno(solution,pred1,pred2,pred3, labels):
    fuzzymeasures = np.array([solution[0],solution[1],solution[2]])
    l = Symbol('l', real = True)
    lam = solve(  ( 1 + l* fuzzymeasures[0]) * ( 1 + l* fuzzymeasures[1]) *( 1 + l* fuzzymeasures[2]) - (l+1), l )
    if len(lam) < 3:
      lam = np.asarray(lam)
    else:
      if lam[0] == 0:
          lam = lam = np.asarray(lam[2])
      elif lam[1] == 0:
          lam = np.asarray(lam[2])
      elif lam[2] == 0:
          lam = np.asarray(lam[1])

    Ypred_fuzzy = np.zeros(shape = pred1.shape, dtype = float)
    for sample in range(0,pred1.shape[0]):
        for classes in range(0,2):
            scores = np.array([pred1[sample][classes],pred2[sample][classes],pred3[sample][classes]])
            permutedidx = np.flip(np.argsort(scores))
            scoreslambda = scores[permutedidx]
            fmlambda = fuzzymeasures[permutedidx]
            ge_prev = fmlambda[0]
            fuzzyprediction = min((scoreslambda[0], fmlambda[0]))
            for i in range(1,3):
                ge_curr = ge_prev + fmlambda[i] + lam * fmlambda[i] * ge_prev
                fuzzyprediction = max((fuzzyprediction,min((scoreslambda[i],ge_curr))))
                ge_prev = ge_curr
            Ypred_fuzzy[sample][classes] = fuzzyprediction
    ypred_fuzzy = np.argmax(Ypred_fuzzy, axis=1)
    pred_label = []
    for i in ypred_fuzzy:
      label = np.zeros((2))
      label[i] = label[i]+1
      pred_label.append(label)
    pred_label = np.array(pred_label)
    acc = accuracy_score(labels,pred_label)
    #print(acc)
    return -acc