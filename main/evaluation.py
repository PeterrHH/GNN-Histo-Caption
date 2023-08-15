# -*- coding=utf-8 -*-
# author: w61
# Test for several ways to compute the score of the generated words.

'''
Need java 1.8
Package:
pycocoevalcap
'''
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.word_based_scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.word_based_scorers:
            #print('computing %s score...'%(scorer.method()))
            print(scorer)
            print(method)
            score, scores = scorer.compute_score(self.gt, self.ref)
    
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("----------Bleu----------")
                    print("%s: %0.3f"%(m, sc))
                total_scores["Bleu"] = score
            else:
                print("----------Other----------")
                print("%s: %0.3f"%(method, score))
                total_scores[method] = score
            print(score)
        
        print('*****DONE*****')
        print(total_scores)
        # for key,value in total_scores.items():
        #     print('{}:{}'.format(key,value))

if __name__ == '__main__':
    ref = {
        '1':['go down the stairs all the way and stop at the bottom .'],
        '2':['this is a cat.'],
        '3':['I know how good this is']
    }
    gt = {
        '1':['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        '2':['It is a cat.','There is a cat over there.','cat over there.'],
        '3':['i know it, you know it','I do now know what this mean','THis is good']
    }
    # 
    # 注意，这里如果只有一个sample，cider算出来会是0，详情请看评论区。
    scorer = Scorer(ref,gt)
    scores = scorer.compute_scores()
    print("-------")
    print(scores)
