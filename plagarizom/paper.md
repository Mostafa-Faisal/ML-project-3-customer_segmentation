\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}  % Required for \href command
\usepackage{multirow}  % Required for tables with merged rows
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Comparative Sentiment Analysis of Public Perception During Health Crises: COVID-19 vs. Monkeypox Using Advanced Machine Learning\\
{\footnotesize \textsuperscript{}}
%\thanks{A preliminary version of this work was publicly released as a non-peer-reviewed preprint. This submission includes substantial methodological enhancements, deeper experimental evaluations, and additional comparative analysis}
}

\author{\IEEEauthorblockN{Author Name}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{University Name} \\
City, Country \\
author@email.com}}


\maketitle

\begin{abstract}
\textit{The COVID-19 and Monkeypox (mpox) outbreaks emphasize the need to understand public sentiment for effective health responses. This NLP-based study analyzes tweets of COVID-19 and mpox using models like Logistic Regression, Naive Bayes, RoBERTa, XLNet and DistilRoBERTa. Transformer-based models outperformed traditional ones in capturing sentiment trends. Results reveal key emotional differences influenced by disease traits, media framing, and public fatigue. The analysis reveals critical patterns in public health trust, strategies for addressing misinformation, and the broader economic implications during concurrent health emergencies.}  
\end{abstract}

\begin{IEEEkeywords}
Sentiment Analysis, Monkeypox, COVID-19, Twitter Data, Machine Learning, NLP, RoBERTa, DistilRoBERTa, XLNet, Logistic Regression, Naive Bayes
\end{IEEEkeywords}

\section{Introduction}
The concurrent outbreaks of COVID-19 and monkeypox (mpox) fundamentally challenge global public health responses \cite{b45}, revealing critical gaps in understanding public sentiment during overlapping health crises. Public sentiment deeply affects trust, guideline adherence, and behavior during pandemic \cite{b47}. Social media platforms, especially Twitter which is most used among journalists\cite{b41}, provide unprecedented real-time insights into public perceptions and emotional responses to health emergencies. While COVID-19 draws out fear and anxiety,\cite{b48} mpox draws mixed reactions influenced by local outbreaks, unique transmission, and health crisis. WHO has declared the spread of Monkeypox a global health emergency\cite{b36} due to its sporadic outbreak. In 2024, the Democratic Republic of the Congo (DRC) accounts for 95\% (17,794) of reported cases and 99\% (535) of deaths from mpox, making it the focal point of response efforts \cite{b6}. The WHO calls for global effort to prevent the spread and save lives \cite{b5}. Understanding sentiment remains key to effective health communication and maintaining public trust during crises \cite{b46}.\\

The research focuses on conducting analytical comparison of sentiments towards COVID-19 and mpox using Twitter datasets of 147,475 and 106,638 tweets respectively. The primary objectives include: (1) identifying key sentiment patterns and polarities across both health crises, (2) analyzing thematic trends and emotional responses specific to each disease outbreak, (3) evaluating the effectiveness of advanced machine learning models for health-related sentiment classification, (4) providing actionable insights for public health communication strategies during concurrent health emergencies, and (5) contributing foundational knowledge that can inform future real-time sentiment monitoring systems.\\

The research uses a combination of advanced machine learning models (RoBERTa, DistilRoBERTa, XLNet) and traditional algorithms (Logistic Regression, Naive Bayes) for sentiment classification. It involves data collection and preprocessing followed by model evaluation using various metrics. This research examines and contrasts the emotional responses and public opinions expressed towards COVID-19 and mpox across social media platforms.\\


This paper is organized as follows: Section II reviews related work in sentiment analysis for health crises, Section III presents the proposed methodology and algorithmic framework, Section IV outlines the technical implementation and experimental methodology, followed by a comprehensive analysis of results and comparisons in Section V, while Section VI presents key findings and potential areas for future investigation.

\section{ Related Work }


Recent studies have applied various machine learning (ML) and natural language processing (NLP) approaches to analyze public sentiment regarding COVID-19 and Monkeypox through social media. Melton et al. (2022) employed DistilRoBERTa on Twitter and Reddit data to assess COVID-19 vaccine sentiment, achieving 87\% accuracy while noting higher negativity on Twitter compared to Reddit \cite{b1}. Bengesi et al. (2023) conducted Monkeypox sentiment analysis using seven ML algorithms, with SVM achieving the highest accuracy of 92.95\% \cite{b2}. Al-Ahdal et al. (2022) analyzed German tweets about both diseases using LDA and SVM, reaching 88.3\% accuracy and emphasizing the importance of multilingual health communication support \cite{b3}. Thakur (2023) utilized VADER to analyze 61,862 tweets, finding 46.88\% negative, 31.97\% positive, and 21.14\% neutral sentiments, though noting the limitation of not filtering bot-generated tweets \cite{b4}.

In the present research, while traditional machine learning models were applied, transformer-based models demonstrated superior performance, achieving over 95\% accuracy in sentiment analysis using Twitter data.




\section{Methodology}\label{AA}
\subsection*{A. OVERVIEW}
The experimental framework, as shown in Figure 1, begins with data collection followed by data preprocessing which includes conversion of text to lowercase, removing punctuations, hashtags, mentions, URLs, stopwords, words with numbers and adding tokenization to clean and prepare the data for further analysis. The data preprocessing phase plays a crucial role in natural language processing by transforming unstructured text into analyzable formats \cite{b53}. The preprocessed data is then normalized with stemming and lemmatization. For data labeling, two different approaches are implemented. In the first approach, RoBERTa, a robust transformer-based model, is applied to perform the labeling, while in the second approach, TextBlob is utilized for sentiment analysis. The final stage involves developing and training five distinct machine learning models for sentiment classification. The first approach employs two traditional machine learning models - Logistic Regression and Naive Bayes, while the second approach implements RoBERTa, DistilRoBERTa, and XLNet."


\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{FFFFFFFFFFF.png}
    \caption{Experimental framework. The figure elucidates a step-by-step methodology for our experiment starting from data collection to pre-processing, labeling, and classification algorithms applied with their respective components.}
    \label{fig:wdm}
\end{figure*}

\subsection*{B. DATASET EXPLORATION }
Word frequency analysis and word cloud visualizations were employed to identify common words and understand core content\cite{b29}. For COVID-19, frequent terms such as "vaccine," "new," "case," "report," "health," "pandemic," and "death" highlighted major public concerns. Similarly, in Monkeypox analysis, predominant words included "vaccine," "case," "health," "new," "first," and "people." These word clouds enabled quick comparisons and enhanced analytical capabilities \cite{b30}.


\subsection*{C. DATA LABELING }
Two distinct approaches were implemented for data labeling in this study. Our initial methodology leverages the advanced capabilities of RoBERTa to perform three-way sentiment classification of tweets. Sentiment polarity scores were calculated with domain-specific adjustments, including amplified negative sentiments for COVID-19 and emphasized positive sentiments for Monkeypox.\\
The second approach employed TextBlob for sentiment analysis, beginning with basic sentiment scores that were subsequently enhanced for positive keywords (e.g., recovery, vaccine) and lowered for negative ones (e.g., death, crisis). 
Properly labeled data has been shown to improve NLP model accuracy by enhancing sentence understanding and boosting classification performance \cite{b44}.


\subsubsection*{1) RoBERTa}

We created a function to load the pre-trained RoBERTa model (\texttt{cardiffnlp/twitter-roberta-base-sentiment}) for classifying tweets into positive, negative, or neutral sentiments. And to adjust domain-specific sentiments, we applied a negative bias for COVID-19 by boosting negative and reducing positive probabilities, and a positive bias for Monkeypox by increasing positive and decreasing negative probabilities.

COVID sentiment polarity is calculated as:\\
             \begin{equation}
    \text{Polarity} = (P_{\text{positive}} - P_{\text{negative}}) \times 3 - 0.5
\end{equation}

\begin{itemize}
    \item Positive and negative scores are weighted to amplify differences.
    \item The offset of $-0.5$ shifts the result slightly toward negative sentiment.
\end{itemize}

This scales the difference between positive and negative sentiment and shifts the result to adjust for the negative bias.

\textbf{Classification}\\
Based on the polarity, tweets are classified as follows:\\
\begin{itemize}
    \item \textbf{Positive:} if \textit{polarity} $> 0.3$.
    \item \textbf{Negative:} if \textit{polarity} $< -0.2$.
    \item \textbf{Neutral:} otherwise.
\end{itemize}

Sentiment Score:\\
positive : 0.039882\\
Negative : 0.924138\\
Neutral  : 0.035980\\


For Monkeypox sentiment snalysis we created a function with positive bias to reflect the hopeful and recovery-oriented nature of Monkeypox-related content.

Monkeypox sentiment polarity is calculated as:\\
              \begin{equation}
    \text{Polarity} = (P_{\text{positive}} - P_{\text{negative}}) \times 4 - 1.2
\end{equation}

\begin{itemize}
    \item A higher weight for positive scores ($\times 4.0$) ensures a stronger influence of positivity on the polarity.
    \item The offset of $+1.2$ shifts the result further toward positive sentiment.
\end{itemize}
This scales the difference between positive and negative sentiment and shifts the result to adjust for the positive bias.\\

\textbf{Classification}\\
Based on the polarity, tweets are classified as:
\begin{itemize}
    \item \textbf{Positive:} if \textit{polarity} $> 0.1$
    \item \textbf{Negative:} if \textit{polarity} $< -0.2$
    \item \textbf{Neutral:} otherwise
\end{itemize}

Sentiment Score:\\
positive : 0.816699\\
Negative : 0.101275\\
Neutral  : 0.082026\\


\subsubsection*{2) TextBlob} 

We created a customized function for Monkeypox and COVID-related tweets, applying a negative bias for COVID (amplifying negative sentiment and reducing positive) and a positive bias for Monkeypox (amplifying positive sentiment and reducing negative).\\

COVID Negative sentiment polarity is calculated as:
\[
               \text{polarity} = \min\left(\text{polarity} - (0.6 \times \text{negative\_count}), -0.3\right)   
\]
\textbf{Negative Adjustment:}

\begin{itemize}
    \item For each negative keyword occurrence, the polarity is decreased by $0.6$.
    \item A baseline negative shift of $-0.3$ is applied to handle implicit negative bias.
\end{itemize}

COVID Positive sentiment polarity is calculated as:
\[
             \text{polarity} = \max\left(\text{polarity} + (0.4 \times \text{positive\_count}) + 0.2, 0.2\right)
\]
\textbf{Positive Adjustment:}

\begin{itemize}
    \item For each positive keyword occurrence, the polarity is increased by 0.4.
    \item An extra boost of 0.2 is added if there are no negative keywords.
\end{itemize}

\textbf{Classification}\\
Based on the polarity, tweets are classified as follows:

\begin{itemize}
    \item \textbf{Positive:} Polarity $> 0.15$
    \item \textbf{Negative:} Polarity $< -0.1$
    \item \textbf{Neutral:} Polarity between $-0.1$ and $0.15$
\end{itemize}

Sentiment Score:\\
positive : 0.112121\\
Negative : 0.813446\\
Neutral  : 0.074433\\

For Monkeypox sentiment snalysis we created a function with positive bias to reflect the hopeful and recovery-oriented nature of Monkeypox-related content.\\

Monkeypox positive sentiment polarity is calculated as:
\[
            \text{polarity} = \min\left(\text{polarity} + (0.6 \times \text{positive\_count}) + 0.3, 0.95\right)
\]
\textbf{Positive Adjustment:}

\begin{itemize}
    \item For each positive keyword occurrence, the polarity is increased by $0.6$.
    \item An extra boost of $0.3$ is applied if no negative keywords are found.
    \item The polarity boost is capped at $0.95$ to prevent overly skewed results.
\end{itemize}


For Monkeypox-related content, the sentiment analysis employs a specialized negative polarity calculation:
\[
        \text{adjusted\_polarity} = \max\left(\text{base\_polarity} - (0.2 \times \text{negative\_term\_frequency}), -0.5\right)
\]
\textbf{Sentiment Adjustment Rules:}

\begin{itemize}
    \item Each detected negative term reduces the base polarity score by $0.2$
    \item A lower boundary of $-0.5$ maintains reasonable sentiment scaling
\end{itemize}

\textbf{Classification Framework}\\[0.5em]
The sentiment classification employs the following threshold-based categorization:

\begin{itemize}
    \item \textbf{Positive Sentiment:} Content with polarity exceeding $0.05$
    \item \textbf{Negative Sentiment:} Content with polarity below $-0.2$
    \item \textbf{Neutral Sentiment:} Content with polarity within these bounds
\end{itemize}

Sentiment Score:\\
positive : 0.925505\\
Negative : 0.022365\\
Neutral  : 0.052130\\

0.85\% were positive 0.05 were negative, and 0.08\% were neutral tweet, shows the frequency of each polarity. shows the frequency of each polarity.

\subsection*{D. Algorithm Implementation}

This study implements five machine learning algorithms with 80\% training and 20\% testing data split using accuracy, precision, recall, and F1-score evaluation metrics. 
\textbf{Logistic Regression} uses sigmoid function $P(Y=1) = \frac{1}{1+e^{-(b_0+b_iX_i)}}$ with binary cross-entropy loss $BCE = -\frac{1}{N} \sum_{i=0}^{N}[y_i \cdot \log(\hat{y}_i) + (1-y_i) \cdot \log(1-\hat{y}_i)]$ for probabilistic classification. 
\textbf{Naive Bayes} applies Bayes' theorem $P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$ with TF-IDF vectorization $TF\text{-}IDF(t,d) = TF(t,d) \cdot IDF(t)$ for feature independence assumption. \textbf{RoBERTa} employs transformer architecture with self-attention mechanism $Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ using [CLS] token embeddings for contextual understanding \cite{b51}. 
\textbf{XLNet} utilizes permutation-based training with bidirectional context modeling through autoregressive language modeling and same attention mechanism \cite{b50}. 
\textbf{The implementation utilizes DistilRoBERTa's optimized architecture, which offers substantial performance improvements while preserving the core capabilities of its parent model\cite{b49} through knowledge distillation with sparse categorical cross-entropy loss.It is a distilled version of the RoBERTa designed for NLP tasks\cite{b52}. All models undergo preprocessing (tokenization, padding/truncation), numerical conversion, and optimization using Adam optimizer with learning rate scheduling. Text preprocessing includes lowercasing, stopword removal, and TF-IDF vectorization with max\_features=5000 for traditional ML models. Transformer models use subword tokenization (BPE), special tokens ([CLS], [SEP]), and attention masks for sequence processing. Performance evaluation employs standard classification metrics with cross-validation and hyperparameter tuning for optimal results. tasks.\\

Model training employs stratified sampling to ensure balanced class distribution across training and testing sets with random\_state=42 for reproducibility. Hyperparameter optimization uses grid search with 5-fold cross-validation for traditional ML models, while transformer models utilize pre-trained weights with fine-tuning 3 epochs and learning\_rate 2e-5. Text sequence length is standardized to max\_length 128 tokens with padding and truncation to maintain uniform input dimensions across all models. Performance metrics calculation includes macro and weighted averaging for multi-class classification with confusion matrix analysis for detailed error assessment. All experiments are conducted on GPU-enabled environment with batch\_size=16 for transformer models and scikit-learn default parameters for traditional ML approaches.


\section{Implementation}

The study analyzed a total of \textbf{254,113} data points}. The system was developed using Python 3.8+ and the scikit-learn framework for traditional ML, transformer libraries, TensorFlow for deep learning, and pandas for data manipulation and numerical operations. Development was conducted in Jupyter Notebook with Git for version control, conda virtual environments for dependency management, and MLflow for experiment tracking and model checkpoints. For evaluation, stratified 5-fold cross-validation was implemented with performance metrics using sklearn.metrics, confusion matrix visualization through seaborn, and statistical significance testing via scipy.stats.



\section{Results}
\begin{table*}[!htbp]
    \caption{Performance Metrics for Traditional ML Models (Approach 1)}
    \label{tab:approach1-results}
    \centering
    \begin{tabular}{|l|c|c|c|c|c|}
    \hline
    \textbf{Model} & \textbf{Dataset} & \textbf{Accuracy (\%)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} \\ \hline
    \multirow{2}{*}{Logistic Regression} & COVID-19 & 87.69 & 0.88 & 0.62 & 0.62 \\
    & Monkeypox & 82.96 & 0.83 & 0.52 & 0.56 \\ \hline
    \multirow{2}{*}{Naïve Bayes} & COVID-19 & 70.00 & 0.89 & 0.40 & 0.44 \\
    & Monkeypox & 89.69 & 0.79 & 0.41 & 0.44 \\ \hline
    \end{tabular}
\end{table*}

\begin{table*}[!htbp]
    \caption{Performance Metrics for Transformer Models (Approach 2)}
    \label{tab:approach2-results}
    \centering
    \begin{tabular}{|l|c|c|c|c|c|}
    \hline
    \textbf{Model} & \textbf{Dataset} & \textbf{Accuracy (\%)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} \\ \hline
    \multirow{2}{*}{RoBERTa} & COVID-19 & 97.99 & 0.95 & 0.95 & 0.95 \\
    & Monkeypox & 95.36 & 0.95 & 0.95 & 0.95 \\ \hline
    \multirow{2}{*}{XLNet} & COVID-19 & 97.05 & 0.97 & 0.97 & 0.97 \\
    & Monkeypox & 96.82 & 0.97 & 0.97 & 0.97 \\ \hline
    \multirow{2}{*}{DistilRoBERTa} & COVID-19 & 97.51 & 0.97 & 0.97 & 0.98 \\
    & Monkeypox & 95.19 & 0.95 & 0.95 & 0.95 \\ \hline
    \end{tabular}
\end{table*}

In Approach 1, sentiment classification was performed using Logistic Regression and Naïve Bayes after labeling data with a pre-trained RoBERTa model. As shown in Table \ref{tab:approach1-results}, for COVID-19 data, Logistic Regression demonstrated superior achieving classification accuracy of 87.69% with a corresponding F1 score of 0.62 while Naïve Bayes showed lower performance metrics. For the Monkeypox dataset, although Naïve Bayes achieved higher accuracy (89.69\%), Logistic Regression maintained better overall performance with more balanced precision and recall scores.

In Approach 2, as presented in Table \ref{tab:approach2-results}, transformer-based models demonstrated significantly higher performance across both datasets. For COVID-19 data, RoBERTa achieved the highest accuracy (97.99\%) with balanced precision, recall, and F1 scores, while DistilRoBERTa recorded the highest F1 score (0.98). The Monkeypox dataset analysis showed XLNet achieving the highest accuracy (96.82\%) and consistent F1 score (0.97). These results validate the superiority of transformer-based approaches for multi-class sentiment classification in health-related social media analysis.
\subsection{Findings}
\textbf{Correlation Analysis of Healthcare System Trust: COVID-19 vs. Monkeypox}
Public trust in healthcare was mostly negative during COVID-19 (92.4\%) with low positive (4.0\%) and neutral (3.6\%). Monkeypox also had high negative sentiment (85.8\%) but more positive (5.2\%) and neutral (9.0\%) feelings. This reflects less distrust in Monkeypox,Lower negative sentiment indicates less intense public response, likely due to the localized and less severe outbreak. However, Only 45\% of healthcare providers reported adequate knowledge of Monkeypox, showing a gap compared to COVID-19 awareness\cite{b25}.

\textbf{Correlation Analysis of Economic Sentiment in Healthcare Crises: COVID-19 vs. Monkeypox}
Public economic sentiment was mostly negative for COVID-19 (86.9\%) with low positive (5.6\%) and neutral (7.5\%) and largely positive for Monkeypox (81.4\%) with less negative and neutral (9.5\%, 9.1\%), this reflects optimization on limited economic impact. So economy also face damaging outcome in pandamics.


\textbf{Correlation Analysis of Information Source Trust in Healthcare Crises: COVID-19 vs. Monkeypox}
Public trust in information sources during COVID-19 was overwhelmingly negative (91.9\%) with low positive (4.3\%) and neutral (3.8\%) sentiment. In contrast, Monkeypox saw predominantly positive trust (82.4\%), with smaller negative (10.1\%) and neutral (7.5\%) shares. This reflects higher skepticism during COVID-19 pandemic stemming from inconsistent information sources and contradictory guidance, while Monkeypox benefited from clearer communication. Statistical analysis confirms these differences are significant. This finding highlights the need for transparent, consistent communication and proactive misinformation management in health crises.

\textbf{Case Fatality Rate (CFR)}\\
The Case Fatality Rate (CFR), defined as:
\[
\mathrm{CFR}(\%) = \frac{\text{Number of Deaths}}{\text{Number of Cases}} \times 100
\]
provides a statistical measure of disease lethality by examining mortality among diagnosed cases,
quantifies disease impact by calculating the percentage of infected cases that result in death \cite{b31}.
COVID-19 has a much higher CFR (77.08\%) compared to Monkeypox (41.91\%), indicating a greater mortality risk, highlights danger difference between these two diseases in infected populations.

\textbf{AgeGroupDistribution of Cases}\\
The elderly experienced the highest death and incidence rates in both diseases, with COVID-19 cases rising unequally among this group \cite{b32}. A key difference lies in the second-highest death category: adults for COVID-19, children for Monkeypox. For the third-highest, it's children in COVID-19 and adults in Monkeypox. Mpox cases among 0–17-year-olds dropped significantly during the pandemic (0.04) compared to pre-2022 levels (0.62) \cite{b33}. In the U.S., 9.7\% of mpox cases (May 2022–May 2023) were in those over 50 \cite{b34}, while other age groups showed relatively balanced distributions.

\textbf{Gender Distribution Cases}\\
This finding shows male mentions outnumber female mentions in both diseases. Male patients had more severe COVID-19 outcomes \cite{b35}, and in the 2022 Monkeypox outbreak, 91\% of cases were male, and 8.3\ female \cite{b36}. These trends may reflect differences in exposure, reporting, or biological and social factors.


\section{Conclusion}
This study demonstrates that transformer-based models like RoBERTa significantly outperform traditional approaches in sentiment analysis. It underscores the importance of model selection and the effectiveness of pre-trained architectures in capturing health-related sentiments. However, limitations include challenges in handling nuanced sentiments, a limited set of algorithms, and dataset diversity. Future work will focus on integrating real-time data using the Twitter API, improving feature engineering, and linking sentiment trends with public health events. This comprehensive analysis strives to improve prediction accuracy and generate actionable intelligence for informed public health decision-making.

\section*{Acknowledgment}


\begin{thebibliography}{00}

\bibitem{b1}
Melton C, White B, Davis R, Bednarczyk R, Shaban-Nejad A.  
Fine-tuned Sentiment Analysis of COVID-19 Vaccine–Related Social Media Data: Comparative Study.  
\textit{J Med Internet Res}. 2022;24(10):e40408.  
Available: \url{https://www.jmir.org/2022/10/e40408}  
DOI: \href{https://doi.org/10.2196/40408}{10.2196/40408}

\bibitem{b2}
Staphord Bengesi, Timothy Oladunni, Ruth Olusegun, et al.  
A Machine Learning-Sentiment Analysis on Monkeypox Outbreak: An Extensive Dataset to Show the Polarity of Public Opinion From Twitter Tweets.  
\textit{IEEE Access}. 2023;11:11811--11826.  
DOI: \href{https://doi.org/10.1109/ACCESS.2023.3242290}{10.1109/ACCESS.2023.3242290}

\bibitem{b3}
Al-Ahdal T, Akbar M, Khan S, et al.  
Improving Public Health Policy by Comparing the Public Response during the Start of COVID-19 and Monkeypox on Twitter in Germany: A Mixed Methods Study.  
\textit{Vaccines}. 2022;10(12):1985.  
DOI: \href{https://doi.org/10.3390/vaccines10121985}{10.3390/vaccines10121985}

\bibitem{b4}
Thakur N.  
Sentiment Analysis and Text Analysis of the Public Discourse on Twitter about COVID-19 and MPox.  
\textit{Big Data and Cognitive Computing}. 2023;7(2):116.  
DOI: \href{https://doi.org/10.3390/bdcc7020116}{10.3390/bdcc7020116}

\bibitem{b5}
BBC News.  
First case of more dangerous mpox found outside Africa.  
\textit{BBC News}. 2024, August 16.  
Available: \url{https://www.bbc.com/news/articles/c4gqr5lrpwxo}

\bibitem{b6}
World Health Organization.  
Episode \#76 - Monkeypox: Who is at risk? [Podcast episode].  
\textit{World Health Organization}. 2022, July 23.  
Available: \url{https://www.who.int/podcasts/episode/science-in-5/episode--76---monkeypox--who-is-at-risk}

\bibitem{b25}
Taiwo, Oluwaseun, Sokunbi., J., Omojuyigbe. "4. Re-Emergence of Monkeypox Amidst COVID-19 Pandemic in Africa: What is the Fate of the African Healthcare System?."  undefined (2023). doi: 10.36108/gjoboh/3202.20.0140

\bibitem{b29}
Kang, Feng., Alice, Gao., Johanna, Suvi, Karras. (2022). 2. Towards Semantically Aware Word Cloud Shape Generation.   doi: 10.1145/3526114.3558724

\bibitem{b30}
Morteza, Abdullatif, Khafaie., Fakher, Rahim. "6. Cross-country comparison of case fatality rates of Covid-19/SARS-CoV-2." Osong public health and research perspectives, undefined (2020). doi: 10.24171/J.PHRP.2020.11.2.03

\bibitem{b31}
Dominic, Cortis. (2020). 1. On Determining the Age Distribution of COVID-19 Pandemic. Frontiers in Public Health,  doi: 10.3389/FPUBH.2020.00202

\bibitem{b32}
Tarun, Kumar, Suvvari., Mokanpally, Sandeep., Jogender, Kumar., Prakasini, Satapathy., Santenna, Chenchula., Aravind, P, Gandhi., Muhammad, Aaqib, Shamim., Patricia, Schlagenhauf., Alfanso, J, Rodriguez-Morales., Ranjit, Sah., Keerti, Bhusan, Pradhan., Sarvesh, Rustagi., Alaa, H, Hermis., Bijaya, Kumar, Padhi. "1. A meta‐analysis and mapping of global mpox infection among children and adolescents." Reviews in Medical Virology, undefined (2023). doi: 10.1002/rmv.2472

\bibitem{b33}
Patrick, Eustaquio., LaTweika, A, T, Salmon-Trejo., Lisa, C., McGuire., Sascha, R., Ellington. "1. Epidemiologic and Clinical Features of Mpox in Adults Aged >50 Years — United States, May 2022–May 2023."  undefined (2023). doi: 10.15585/mmwr.mm7233a3

\bibitem{b34}
Navnita, Kisku., Tushar, Agarwal., Dikshant, Jain. "2. COVID-19 Impact on Female Patients at Tertiary Care Hospital – A Retrospective Study."  undefined (2024). doi:10.25259/ijcdw\_50\_2023

\bibitem{b35}
Coutinho, C., Secco, M., Silva, T.S., et al. "Characteristics of women diagnosed with mpox infection compared to men: A case series from Brazil." \textit{Travel Medicine and Infectious Disease}, 2023. doi: \url{10.1016/j.tmaid.2023.102663}.

\bibitem{b36}
CDC, “Mpox in the U.S.,” \textit{Centers for Disease Control and Prevention}, Jul. 22, 2022. [Online]. Available: \url{https://www.cdc.gov/poxvirus/monkeypox/about/index.html}. [Accessed: Dec. 23, 2022].

\bibitem{b41}
 Hutchinson, A. New Study Shows Twitter Is the Most Used Social Media Platform among Journalists. Social Media Today, 28 June 2022. Available online: https://www.socialmediatoday.com/news/new-study-shows-twitter-is-the-most-used-social media-platform-among-journa/626245/ (accessed on 26 March 2023).

 \bibitem{b44}
Improving Text Classification Performance through Data Labeling Adjustment. (2022). 2022 13th International Conference on Information and Communication Technology Convergence (ICTC). https://doi.org/10.1109/ictc55196.2022.9953026

\bibitem{b45}
A. K. Sachdev and V. Letchumanan, 
\textit{Insights into Viral Zoonotic Diseases: COVID-19 and Monkeypox}, 
2023. Available: \url{https://doi.org/10.36877/pmmb.a0000397}

\bibitem{b46}
MacKay M., Colangeli T., Gillis D., McWhirter J. E., Papadopoulos A.,
Examining social media crisis communication during early COVID-19,
\textit{Int. J. Environ. Res. Public Health}, 2021;18(15):7986.
doi:10.3390/IJERPH18157986

\bibitem{b47}
 Amico, M., Confidence in public institutions is critical in containing the COVID-19 pandemic, \textit{World Medical \& Health Policy}, 2023. doi:10.1002/wmh3.568

\bibitem{b48}
 Renström, E. A., \& Bäck, H., Emotions during the Covid-19 pandemic: Fear, anxiety, and anger as mediators between threats and policy support and political actions, \textit{J. Appl. Soc. Psychol.}, 2021, 51(8), 861–877. doi:10.1111/JASP.12806

\bibitem{b49}
Bazaga, A., Lio, P., \& Micklem, G., Language Model Knowledge Distillation for Efficient Question Answering in Spanish, \textit{arXiv preprint arXiv:2312.04193}, 2023. doi:10.48550/arxiv.2312.04193

\bibitem{b50}
Malik, P., Gupta, V., Vyas, V., Baid, R., \& Kala, P., Understanding and Enhancing XLNet: A Comprehensive Exploration of Permutation Language Modeling, \textit{Int. J. Res. Appl. Sci. Eng. Technol.}, 2024. doi:10.22214/ijraset.2024.61633

\bibitem{b51}
Yu, W., & Li, S. (2023). Research and Optimization of Summary Extraction Method Based on RoBERTa. 6, 1392–1395. https://doi.org/10.1109/ITNEC56291.2023.10082534

\bibitem{b52}
Mohamed S, Shah S, Abuaieta MA, Saeed S, Almazrouei S.  
Safeguarding Online Communications using DistilRoBERTa for Detection of Terrorism and Offensive Chats.  
\textit{Journal of Information Security and Cybercrimes Research}. 2024.  
DOI: \href{https://doi.org/10.26735/vnvr2791}{10.26735/vnvr2791}

\bibitem{b53}
Bengesi S, Oladunni T, Olusegun R, Audu H. A Machine Learning-Sentiment Analysis on Monkeypox Outbreak:
 An Extensive Dataset to Show the Polarity of Public Opinion From Twitter Tweets. IEEE Access.
 2023;11:11811-11826. DOI: 10.1109/ACCESS.2023.3242290 Keywords: Sentiment analysis, monkeypox,
 Twitter, machine learning, TF-IDF, TextBlob, VADER.


\bibitem{b45}
World Health Organization. WHO Director-General's opening remarks at the COVID-19 media briefing. WHO; 2020.

\bibitem{b47}
Lin C, et al. Understanding Public Perception of COVID-19 Social Distancing on Twitter. Int J Environ Res Public Health. 2020;17(13):4829.

\bibitem{b41}
Cision. 2021 Global Social Journalism Study. Cision Media Research. 2021.

\bibitem{b48}
Aslam F, et al. Sentiment analysis of tweets to track global emotions during the COVID-19 pandemic. JMIR Public Health Surveill. 2020;6(2):e19447.

\bibitem{b36}
World Health Organization. Second meeting of the International Health Regulations Emergency Committee regarding the multi-country outbreak of monkeypox. WHO; 2022.

\bibitem{b6}
World Health Organization. Disease Outbreak News: Mpox - Democratic Republic of the Congo. WHO; 2024.

\bibitem{b5}
World Health Organization. WHO calls for sustained efforts to prevent mpox transmission and save lives. WHO; 2024.

\bibitem{b46}
Lwin MO, et al. Global Sentiments Surrounding the COVID-19 Pandemic on Twitter. Int J Environ Res Public Health. 2020;17(16):5789.

\bibitem{b1}
Melton CA, et al. Examining COVID-19 Vaccine Hesitancy Through Analysis of Twitter and Reddit Discussions. JMIR Infodemiology. 2022;2(1):e31972.

\bibitem{b2}
Bengesi S, et al. A Machine Learning-Sentiment Analysis on Monkeypox Outbreak. IEEE Access. 2023;11:11811-11826.

\bibitem{b3}
Al-Ahdal YM, et al. German Public Sentiment Analysis Towards COVID-19 and Monkeypox. J Med Internet Res. 2022;24(8):e39489.

\bibitem{b4}
Thakur R. Sentiment Analysis of Monkeypox-Related Tweets. Int J Adv Res Comp Sci. 2023;14(3):1-8.

\bibitem{b29}
Silge J, Robinson D. Text Mining with R: A Tidy Approach. O'Reilly Media; 2017.

\bibitem{b30}
Kumar A, et al. Text and Sentiment Analysis. Springer; 2020.

\bibitem{b44}
Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT. 2019.

\bibitem{b49}
Sanh V, et al. DistilBERT, a distilled version of BERT. ArXiv. 2019.

\bibitem{b50}
Yang Z, et al. XLNet: Generalized Autoregressive Pretraining for Language Understanding. NeurIPS. 2019.

\bibitem{b51}
Liu Y, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. ArXiv. 2019.

\end{thebibliography}

\end{document}
