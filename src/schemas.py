import pandas as pd
from pydantic import BaseModel, Field
from datetime import date, time
from typing import Optional, Literal
from uuid import UUID


class TransactionData(BaseModel):
    transaction_id: Optional[UUID] = Field(None, alias="Transaction ID")
    date_of_purchase: Optional[date] = Field(None, alias="Date of Purchase")
    time_of_purchase: Optional[time] = Field(None, alias="Time of Purchase")
    purchase_type: Literal["Online", "Station"] = Field(None, alias="Purchase Type")
    payment_method: Literal["Contactless", "Credit Card", "Debit Card"] = Field(
        None, alias="Payment Method"
    )
    railcard: Literal["Adult", "None", "Disabled", "Senior"] = Field(
        None, alias="Railcard"
    )
    ticket_class: Literal["Standard", "First Class"] = Field(None, alias="Ticket Class")
    ticket_type: Literal["Advance", "Off-Peak", "Anytime"] = Field(
        None, alias="Ticket Type"
    )
    price: float = Field(None, alias="Price")
    departure_station: str = Field(None, alias="Departure Station")
    arrival_destination: str = Field(None, alias="Arrival Destination")
    date_of_journey: Optional[date] = Field(None, alias="Date of Journey")
    departure_time: Optional[str] = Field(None, alias="Departure Time")
    arrival_time: str = Field(None, alias="Arrival Time")
    actual_arrival_time: str = Field(None, alias="Actual Arrival Time")
    journey_status: Literal["On Time", "Delayed", "Cancelled"] = Field(
        None, alias="Journey Status"
    )
    reason_for_delay: str = Field(None, alias="Reason for Delay")

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
