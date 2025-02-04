import pandas as pd
from pydantic import BaseModel, Field
from datetime import date, time
from typing import Optional
from uuid import UUID


class TransactionData(BaseModel):
    transaction_id: Optional[UUID] = Field(None, alias="Transaction ID")
    date_of_purchase: Optional[date] = Field(None, alias="Date of Purchase")
    time_of_purchase: Optional[time] = Field(None, alias="Time of Purchase")
    purchase_type: Optional[str] = Field(None, alias="Purchase Type")
    payment_method: Optional[str] = Field(None, alias="Payment Method")
    railcard: Optional[str] = Field(None, alias="Railcard")
    ticket_class: Optional[str] = Field(None, alias="Ticket Class")
    ticket_type: Optional[str] = Field(None, alias="Ticket Type")
    price: Optional[float] = Field(None, alias="Price")
    departure_station: Optional[str] = Field(None, alias="Departure Station")
    arrival_destination: Optional[str] = Field(None, alias="Arrival Destination")
    date_of_journey: Optional[date] = Field(None, alias="Date of Journey")
    departure_time: Optional[str] = Field(None, alias="Departure Time")
    arrival_time: Optional[str] = Field(None, alias="Arrival Time")
    actual_arrival_time: Optional[str] = Field(None, alias="Actual Arrival Time")
    journey_status: Optional[str] = Field(None, alias="Journey Status")
    reason_for_delay: Optional[str] = Field(None, alias="Reason for Delay")

    class Config:
        allow_population_by_field_name = True

    def to_dataframe(self):
        # Convert the Pydantic object to a dictionary
        data_dict = self.model_dump(by_alias=True)

        # Create a DataFrame from the dictionary
        df = pd.DataFrame([data_dict])

        # Remove columns with all None values
        df = df.dropna(axis=1, how="all")

        return df


class InferenceData(BaseModel):
    refund_request: Optional[str] = Field(None, alias="Refund Request")

    class Config:
        allow_population_by_field_name = True
