# DataOrbit 2025 -- CashGPT

## How to run

1) Run the local API: python api.py
2) Submit a POST request to http://127.0.0.1:5000/cashgpt/predict

Example post requests:
```
{
    "complaint_included": false,
    "AGE": 65,
    "INCOME": "$52,125",
    "TRAVTIME": 14,
    "BLUEBOOK": "$25,532",
    "CAR_TYPE": "Sports Car",
    "REVOKED": "No",
    "MVR_PTS": 3,
    "CAR_AGE": 3,
    "URBANICITY": "Highly Urban/ Urban",
    "Duration": 50,
    "Company": "Anthem Health Plans, Inc"
}
```
```
{
    "complaint_included": true,
    "AGE": 37,
    "INCOME": "$37,000",
    "TRAVTIME": 30,
    "BLUEBOOK": "$15,000",
    "CAR_TYPE": "Minivan",
    "REVOKED": "No",
    "MVR_PTS": 1,
    "CAR_AGE": 10,
    "URBANICITY": "Highly Urban/ Urban",
    "Duration": 10,
    "Company": "ACE American Insurance Company",
    "Coverage": "A & H",
    "SubReason": "Claim Delay"
}
```
