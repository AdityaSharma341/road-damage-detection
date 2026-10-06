from sqlalchemy.orm import declarative_base
from sqlalchemy import Column,Integer,String

Base = declarative_base()

class ImageRecord(Base):

    __tablename__="image_records"

    id=Column(Integer,primary_key=True,index=True)

    input_image=Column(String(500))

    output_image=Column(String(500))