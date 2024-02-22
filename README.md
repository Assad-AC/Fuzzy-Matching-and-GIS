
This is still in very early stages of development, but it is an entirely functional proof of concept. This repo hosts a Python Shiny application designed to enable the user to: use fuzzy logic to see the most relevant matching pairs between two datasets, while also using coordinate points to further determine whether the matching pairs represent the same location. This is useful for anyone who needs to pair data with strings which are inexact, especially to develop master lists across heterogeneous datasets. The user has the options to: select and confirm a proposed pairing; tentatively select a pairing; save all options offered by the app for a particular value either to review later or to dismiss as not containing a match; and to skip a value, meaning return to it later by sending it to the back of the queue of values to consider. The user's progress can be saved by clicking to exporting a JSON file, and it can be loaded by importing said JSON file.

The screenshot below shows the application prior to clicking "run"; more detailed screenshots may follow with the use of non-sensitive mock data.
![image](https://github.com/Assad-AC/Fuzzy-Matching-and-GIS/assets/126238295/79f2a146-356f-43ae-97c5-b690fa2fc847)


Pending development tasks:
- Speed optimisation
- Error handling
- Refactoring
- Modularisation
- UI options to switch on or off certain features
