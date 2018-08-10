DeepMatch 
=========

DeepMatch provides a toolkit and benchmarks for sentence matching, whose inputs are two sentences and output the relation of them. Deepmatch implements a number of state-of-the-art approaches for this goal. By applying Deepmatch, users can judge whether two sentences mean the same.

Benchmarks
----------

+-------------------------------+-----------------------+-----------------------+-----------------------+
|      Models                   |        Quora          |         MSR           |         ATEC          |
+                               +-----------+-----------+-----------+-----------+-----------+-----------+
|                               |  Accuracy |    F1     |  Accuracy |    F1     |  Accuracy |    F1     |
+===============================+===========+===========+===========+===========+===========+===========+
|     Siamese-CNN [1_]          |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
| LSTM-angel-distance [#r1]_    |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     LSTM-concat [#r1]_        |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     DSSM [#r2]_               |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     CDSSM [#r3]_              |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     LR-CDNN [#r5]_            |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
| Decomposable-attention [#r6]_ |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     DRMM [#r7]_               |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     ABCNN [#r8]_              |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     BiMPM [#r9]_              |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+
|     DRCN [#r10]_              |           |           |           |           |           |           |
+-------------------------------+-----------+-----------+-----------+-----------+-----------+-----------+


.. _1:

Publications
------------

[#] Nikhil Dandekar. https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning

[#] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry P. Heck. `Learning Deep Structured Semantic Models for Web Search using Clickthrough Data <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf>`_, **CIKM**, 2013

[#] Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, Grégoire Mesnil. `Learning Semantic Representations using Convolutional Neural Networks for Web Search <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf>`_, **WWW**, 2014

[#] Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning. `A Large Annotated Corpus for Learning Natural Language Inference <https://arxiv.org/pdf/1508.05326>`_, **EMNLP**, 2015

[#] Aliaksei Severyn, Alessandro Moschitti. `Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks <http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf>`_, **SIGIR**, 2015

[#] Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. `A Decomposable Attention Model for Natural Language Inference <https://arxiv.org/pdf/1606.01933.pdf>`_ , **EMNLP**, 2016

[#] Jiafeng Guo, Yixing Fan, Qingyao Ai, W. Bruce Croft. `A Deep Relevance Matching Model for Ad-hoc Retrieval <https://arxiv.org/pdf/1711.08611>`_, **CIKM**, 2016

[#] Wenpeng Yin, Hinrich Schütze, Bing Xiang, Bowen Zhou. `ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs <https://arxiv.org/pdf/1512.05193.pdf>`_, **TACL**, 2016

[#] Zhiguo Wang, Wael Hamza, Radu Florian. `Bilateral Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/pdf/1702.03814.pdf>`_, **IJCAI**, 2017

[#] Seonhoon Kim, Jin-Hyuk Hong, Inho Kang, Nojun Kwak. `Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information <https://arxiv.org/pdf/1805.11360>`_, **arXiv**, 2018

