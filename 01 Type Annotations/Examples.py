from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int

movie = Movie(name="The Matrix", year=1999)