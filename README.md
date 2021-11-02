# Machine Translation project: BPE on Inuktitut
Authors: Viktorija Buzaitė, Rafał Černiavski, Laura Janulevičiutė, Eva Elžbieta Sventickaitė




## A comparative study of the effects of BPE on SMT andNMT performance as applied on Inuktitut

The main purpose of this study was to examine the impact of different BPE merges on the performance of four MT models: SMT, two Seq2Seq models, namely LSTM and CNN, as well as the Transformer. For this purpose, the morphologically rich Inuktitut language has been chosen. The results show that no BPE was the best solution for SMT, LSTM performed best with 30,000 BPE merges, whereas CNN and Transformer scored the highest with 10,000 merges. Additionally, unlike SMT and the Transformer, both Seq2Seq models did not show a clear pattern of improvement in regards to the number of BPE merges. Furthermore, the best BLEU score was achieved by the Seq2Seq models, first CNN with LSTM a close second. Nonetheless, the qualitative analysis suggested that, firstly, the Transformer produces more eligible translations, and, secondly, different evaluation metric could be used in future work. It should be added that although the Transformer-based model did not perform particularly well, a possible solution could be reducing model complexity and size. Finally, even though the Transformer is currently regarded in the community as the most efficient architecture, our study has also indicated the value of investigating older models as well. 


## Contents

You can find the results for the CNN model in the ```cnn``` folder.

The report in its entirity can be read in the main repo, ```MT_project_report.pdf```.
