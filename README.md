# Insurance Recommender

## Folder Structure

```
insurance_recommender/
│
├── data/
│   └── synthetic_dataset.csv       
│
├── model/
│   └── insurance_model.pkl         
│   └── label_encoders.pkl         
│
├── generate_data.py               
├── train.py                        
├── recommend.py                    
├── app.py                          
└── requirements.txt               
```

## How to Run

```bash

pip install -r requirements.txt
python generate_data.py
python train.py
python recommend.py
python app.py
```
