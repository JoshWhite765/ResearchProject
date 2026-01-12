This study analyses how public feedback including news articles and social media can seed AI to greatly influence public opinion, in particular the study focuses on patterns that can be found with news articles and social media posts that are believed to be fake. 
Such patterns include key phrases that can mark out articles as being fake or articles that when fact checked from reliable sources are considered fake, or a flood of fake news posts at a particular time e.g. an election or news articular always coming from the same source.
In order to get a balanced view then multiple data sets are used and the data from these passed over multiple different LLMs, analysing the fake news from these to discover the types of patterns that can be found and the prevalence of these patterns.
The benefit of this study should help future AI models determine how to detect fake news more reliably, using the information from patterns detected to learn of the web sites to trust i.e. those used to seed AI, those articles offering a clear gain to a company or political or religious affiliation, awareness of significant events when fake news is more likely to be triggered and patterns with the timing of fake news.

**The folder structure:** 
├── Datasets - The datasets the programs use
│   ├── ISOTFAKE.csv
│   ├── ISOTTrue.csv
│   ├── LIARDATA.tsv
│   ├── LIARDATA.tsv:Zone.Identifier
│   ├── MyOwnFakeAndReal.csv
│   └── ScrapedData.csv
├── Extractors - Python programs to extract fake news phrases off each dataset 
│   ├── FakeNewsDetectorHF.py
│   ├── FakeNewsDetectorISOT.py
│   ├── FakeNewsDetectorLIAR.py
│   ├── FakeNewsDetectorMyOwn.py
│   └── FakeNewsDetectorScraped.py
├── Phrases - The list of phrases extracted by the python programs above 
│   ├── HFPhrases.csv
│   ├── ISOTPhrases.csv
│   ├── LiarPhrases.csv
│   ├── MyOwnPhrases.csv
│   └── ScrapedPhrases.csv
├── Scraper
│   └── Scraper.py - Python program used to scrape data off the science feedback site
├── FakeNewsDistribution - To visualise Fake news frequencies within USA using the LIAR dataset
│   ├── FakeNewsOnUSAMap.png
│   └── VisualiseFakeLocations.py - python program to generate the chloropeth USA map with fake news freq
├── WordClouds - To visualise word frequencies in a word cloud
│    ├── hf_wordcloudBETTER.png
│    ├── ISOT_wordcloudPatterns.png
│    ├── LIAR_wordcloudBETTER.png
│    └── ScrapedData_wordcloudPatterns.png
└── Barcharts - To visualise word frequencies in a bar chart 
    ├── hf_word_freq_barchart.png
    ├── ISOT_word_freq_barchart.png
    ├── LIAR_word_freq_barchart.png
    └── ScrapedData_word_freq_barchart.png
