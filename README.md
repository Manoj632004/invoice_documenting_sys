# invoice documenting system

•	The aim of this project is to automate invoice data extraction <br/>
•	Utilized OCR, appropriate text preprocessing to feed LLM to extract and structure text from invoice images into formatted tables. <br/>
libraries and tech used: PyTorch, pandas, OpenCV, PyTesseract,  GPT4all

invoice image datasets:<br/>
https://machinelearning.inginf.units.it/data-and-tools/ghega-dataset<br/>
https://universe.roboflow.com/jakob-awn1e/receipt-or-invoice/dataset/5



table-transformer-detection from hugging face did not work for table det4ection as most invoicees and recipets dont have defined boundaries of tables

Layout Parser didnt work because depreciated functionalities
