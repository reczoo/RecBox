# DeepCTR

\section{Log Parsing}\label{sec:approach}


\subsection{Overview}

Accuracy

Robustness

Parameter tuning

Efficiency


\subsection{Existing Log Parsing Methods}

Log parsing has been widely studied in recent years. Among
all the approaches proposed, we choose four representative
ones, which are in widespread use for log mining tasks. With
the main focus on evaluations of these log parsing methods,
we only provide brief reviews of them; the details can be found
in the corresponding references.


\subsection{Tool Implementation}
Although automated log parsing has been studied for many years, it is still not a well-received technique in industry. This is largely due to the lack of publicly available tool implementations that are ready for industrial use. For operation engineers who often have little expertise in machine learning, implementing an automated log parsing tool requires non-trivial efforts, which may exceed the overhead for crafting regular expressions. Our project aims to bridge this gap between research and industry and promote the adoption for automated log parsing. We have implemented an open-source log parsing toolkit, namely logparser, and released a large benchmark set as well. Since its first release, logparser and the benchmarks have been requested and used by more than 40 organizations all over the world as of this writing. As a part-time project, the implementations of logparser takes over two years and have 11.7K LOC in Python. Currently, logparser contains a total of 13 log parsing methods proposed by researchers and pracitioners. Among them, four log parsers (SLCT, LogCluster, LenMa, MoLFI) are released by existing research. However, they are implemented in different programming languages and have different input/ouput formats. Examples and documentation are also missing or incomplete, making it difficult for a trial. For ease of use, we define standard input/output interfaces for different log parsing methods and further wrap up the existing tools into a single Python package. Logparser requires a raw log file with free-text log messages as input, and then outputs a file containing structured events and an event template file with aggregated event statistics. The outputs can be easily fed into subsequent log mining tasks. In addition, we provide 16 different types of log datasets with an total amount of 87GB to benchmark the accuracy and efficiency of logparser. Our benchmarking results can help engineers identify the strengths and weakness of different log parsing tools and evaluate their possibility for industrial use cases.

It is also worth noting that our current implementations target at exactly reproducing the log parsing methods described in the original work. They are written in Python and run in a single thread. As we will show in the next section, some parsers such as LKE, LogSig, and LenMa cannot scale well to large log datasets. Although we plan to parallelize their implementations with Spark to handle big data in our future work, users may need to pay more attention when using our current toolkit.











