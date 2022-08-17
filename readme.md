# A System Suitability Testing Platform for Untargeted, High-Resolution Mass Spectrometry


## Abstract

The broad coverage of untargeted metabolomics poses fundamental challenges for the harmonization of measurements along time, even if they originate from the very same instrument. Internal isotopic standards can hardly cover the chemical complexity of study samples. Therefore, they are insufficient for normalizing data a posteriori as done for targeted metabolomics. Instead, it is crucial to verify instrument’s performance a priori, that is before sample are injected. Here we propose a systems suitability testing system for time-of-flight mass spectrometers. It includes a chemically defined quality control mixture, a fast acquisition method, software for extracting ca. 3000 numerical features from profile data, and a simple web-service for monitoring. We ran a pilot for 21 months and present illustrative results for anomaly detection or learning causal relationships between the spectral features and machine settings. Beyond mere detection of anomalies, our results highlight several future applications such as (i) recommending instrument retuning strategies to achieve desirable values of quality indicators, (ii) driving preventive maintenance, or (iii) using the obtained, detailed spectral features for posterior data harmonization.

![Monitoring system snapshot](https://github.com/dmitrav/monitoring_system/blob/master/img/snapshot.png)

The code of the Shiny app can be found [here](https://github.com/zamboni-lab/SST-platform-shiny).
