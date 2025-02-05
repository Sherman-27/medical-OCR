import pandas as pd
import random
from twilio.rest import Client

def send_sms(body, recipient_number="+919867273743"):
    """Send SMS notification to a registered medical reviewer."""
    account_sid = "AC98b13e128c840abae714902f113e932d"
    auth_token = "e0d69d3f952ab4b124c79f1caa64a8f9"
    client = Client(account_sid, auth_token)
    
    try:
        message = client.messages.create(
            body=body,
            from_="+12769001322",
            to=recipient_number
        )
        print("SMS notification sent successfully.")
    except Exception as e:
        print("Failed to send SMS:", e)

def check_medicine_availability(medicine_name, database_file="Datasets\medicines_A_10.csv"):
    """Check if the medicine is available in the CSV database."""
    try:
        df = pd.read_csv(database_file)
        return medicine_name in df["name"].values
    except Exception as e:
        print("Error reading database:", e)
        return False

def scan_prescription(prescription_image):
    """Simulated OCR function to extract medicine names from a prescription."""
    # Random failure simulation (20% chance OCR fails completely)
    if random.random() < 0.2:
        return None  # OCR failed to extract text
    
    extracted_medicines = ["Paracetamol", "Ibuprofen", "Aspirin"]  # Example recognized medicines
    return extracted_medicines if random.random() > 0.3 else []  # Simulated failure to detect medicines

def handle_prescription(prescription_image, database_file="medicine_database.csv"):
    """Process prescription and check medicine availability."""
    extracted_medicines = scan_prescription(prescription_image)
    
    if extracted_medicines is None:
        print("OCR failed to recognize the prescription.")
        send_sms("Prescription OCR Failure: The prescription could not be read. Please review it manually.")
        return
    
    if not extracted_medicines:
        print("No medicines detected in the prescription.")
        send_sms("Prescription Unrecognized: No medicines could be identified. Please review it manually.")
        return
    
    for medicine in extracted_medicines:
        if check_medicine_availability(medicine, database_file):
            print(f"{medicine} is available in the database.")
        else:
            print(f"{medicine} is NOT found in the database. Sending for manual review...")
            send_sms(f"Medicine Not Found: The medicine '{medicine}' is missing. Please verify and update the database.")

# Example Usage
prescription_image = "amoxicillin.png"  # Placeholder image file
handle_prescription(prescription_image, "Datasets\medicines_A_10.csv")

# changes to be made in this file