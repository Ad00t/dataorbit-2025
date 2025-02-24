import React, { useState } from "react";
import { Container, TextField, Button, Typography, Box } from "@mui/material";
import axios from "axios";

export default function HomePage() {
  const [formData, setFormData] = useState({
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
});

  const formToLabel = {
    "complaint_included": "Complaint Included? (true/false)",
    "AGE": "Age",
    "INCOME": "Income",
    "TRAVTIME": "Travel Time (hours)", 
    "BLUEBOOK": "Kelly Blue Book Value",
    "CAR_TYPE": "Car Type",
    "REVOKED": "License Revoked? (Yes/No)",
    "MVR_PTS": "MVR points",
    "CAR_AGE": "Age of Car (years)",
    "URBANICITY": "Urbanicity",
    "Duration": "Duration of Complaint (days)",
    "Company": "Insurance Company",
    "Coverage": "Coverage Type",
    "SubReason": "Complaint Reason"
  }

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/predict",
        formData,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        }
      );
      alert(`P(initial claim approved): ${Math.round(response.data.p_initial_claim_approved * 100, 2)}%\nP(complaint approved): ${Math.round(response.data.p_complaint_approved * 100, 2)}%\nP(finally approved): ${Math.round(response.data.p_final_approved * 100, 2)}%`);
    } catch (error) {
      alert(error);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box mt={5} p={4} boxShadow={3} borderRadius={2}>
        <Typography variant="h4" align="center" gutterBottom>
          CashGPT Insurance Prediction
        </Typography>
        <form onSubmit={handleSubmit}>
          {Object.keys(formData).map((field, index) => (
            <TextField
              key={index}
              name={field}
              label={formToLabel[field]}
              value={formData[field]}
              onChange={handleChange}
              margin="normal"
              fullWidth
              required
            />
          ))}
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 2 }}
          >
            Submit
          </Button>
        </form>
      </Box>
    </Container>
  );
}
