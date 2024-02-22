from shiny import App, render, ui, reactive, req
import pandas as pd
from typing import Callable
import jellyfish as jf
import numpy as np
from itertools import product
import re
from ipyleaflet import Map, Marker, LayerGroup, Layer, CircleMarker, DrawControl, TileLayer, GeoJSON, MarkerCluster
from ipywidgets import HTML
from shinywidgets import output_widget, render_widget
import geopandas as gpd
import json
import os
import time
import asyncio

app_ui = ui.page_fluid(

    ui.panel_title("Fuzzy matching assistant"),

    ui.input_text(id = "AppWd",
                  label = "Copy-paste the path to the application folder",
                  placeholder = "C:\\Users\\username\\Documents\\QC tool"),

    ui.input_file(id = "fileOneToUploadDir", label = "Choose first CSV File", accept=[".csv"], multiple=False),

    ui.input_file(id = "fileTwoToUploadDir", label = "Choose second CSV File", accept=[".csv"], multiple=False),

    ui.input_select("SelectDf_ReceivingVals", "Choose the dataset which will 'receive' matches", choices=[]),

    ui.input_select("SelectCol_ReceivingVals", "Choose the column which will 'receive' matches", choices=[]),

    ui.input_select("SelectDf_CouldAssignVals", "Choose the dataset with values that could be assigned for matching", choices=[]),

    ui.input_select("SelectCol_CouldAssignVals", "Choose the column with the 'could-assign' values", choices=[]),

                            
    ui.input_numeric("NumberOfGroupingPairs", 
                     "Choose the number of (pairs of) columns to group the comparisions by", 
                     value = 0),

    ui.output_ui("GroupingPairs_Ui", "Temporary: type in the grouping pairs in dictionary form, beginning with the receiving dataset side."),

    ui.output_text("UiGroupingPairsDupeWarning"),
    
    ui.input_checkbox_group("OtherUniquifyingCols_ReceivingVals",
                            "Select other columns used to create a unique entry for the 'receiving' dataset",
                            choices = []),

    ui.input_checkbox_group("OtherUniquifyingCols_CouldAssignVals",
                            "Select other columns used to create a unique entry for the 'could-assign' dataset",
                            choices = []),

    ui.row(ui.column(6, ui.input_select("Select_LatCol_Receiving", "Choose the latitude column from the 'receiving' dataset", choices=[])),
           ui.column(6, ui.input_select("Select_LongCol_Receiving", "Choose the longitude column from the 'receiving' dataset", choices=[]))
           ),

    ui.row(ui.column(6, ui.input_select("Select_LatCol_CouldAssign", "Choose the latitude column from the 'could-assign' dataset", choices=[])),
           ui.column(6, ui.input_select("Select_LongCol_CouldAssign", "Choose the longitude column from the 'could-assign' dataset", choices=[]))
           ),

    ui.input_text(id = "ShapeFileToUploadDir", label = "Copy-paste the path to the shapefile folder"),

    ui.input_select(id = "SelectShapeFile", label = "Select Shapefile", choices=[]),
    
    ui.input_checkbox(id = "AutoMatchExactMatches", label = "Automatch when strings are exactly the same",
                      value = False),

    ui.input_numeric(id = "Threshold_ExcludeFuzzyScoresAbove", label = "Exclude scores higher than...", value = 5),

    ui.input_file(id = "UploadDir_ProgressSaveFile", label = "Choose .json Fuzzy Matcher save file", accept = [".json"], multiple = False),

    ui.input_action_button(id = "RunProcess", label = "Run"),

    ui.output_data_frame("DisplayInitial"),

    ui.output_text("CurrentStageTextFeedback"),

    ui.row(
        ui.column(6, ui.output_data_frame("ResponsiveDisplay")),

        ui.column(6, output_widget("AssignmentOptsMap"))                 
                  
                  ),

    ui.input_action_button(id = "ConfirmButton", label = "Confirm", style="background-color: green; color: white;"),

    ui.input_action_button(id = "MarkAsUncertainGuessButton", label = "Confirm as only an uncertain guess", style = "background-color: #90ee90; color: black;"),

    ui.input_action_button(id = "FlagButton", label = "Flag options for further consideration", style = "background-color: orange; color: black;"),

    ui.input_action_button(id = "DismissButton", label = "Dismiss as unmatchable", style = "background-color: red; color: white;"),

    ui.input_action_button(id = "SkipButton", label = "Skip", style = "background-color: blue; color: white;"),

    
    ui.output_data_frame("Display_ValPairAssignmentsMade"),

    ui.download_button(id = "ExportAssignedMatches", label = "Export assigned matches as CSV file"),

    ui.output_data_frame("Display_ValPair_AntiAssignments"),

    ui.download_button(id = "ExportAntiAssignedMatches", label = "Export flagged or dismissed options as CSV file"),


    ui.output_data_frame("Display_ValPairAssignmentsRemaining"),
    
    ui.download_button(id = "ExportRemaining", label = "Export unassigned options as CSV file"),

    ui.download_button(id = "ExportProgressSaveFile", label = "Export matching progress save file")
)



def jf_ElementWiseListPair(jfFunc: Callable, xList: list, yList: list):
    return [jfFunc(xList[i], yList[i]) for i in range(len(xList))]


def fuzzyScoreLists(df_WReceivingVals:pd.DataFrame,
                    col_ReceivingVals:str,
                    df_WCouldAssignVals:pd.DataFrame,
                    col_CouldAssignVals:str,
                    groupingVarsDict:dict):
    
    ReceivingDf_columns_mapping = {}
    # Iterate through columns in df_WReceivingVals
    for col in df_WReceivingVals.columns:
        # Check if the column is in groupingVarsDict
        if col not in groupingVarsDict.keys():
            # If not, rename the column by appending "_ReceivingVals"
            new_col_name = col + "_ReceivingVals"
        else:
            # If it is, keep the original column name
            new_col_name = col
        # Add the mapping to the dictionary
        ReceivingDf_columns_mapping[col] = new_col_name
    
    # Rename columns in df_WReceivingVals using the dictionary
    ReceivingVals_Df = df_WReceivingVals.rename(columns=ReceivingDf_columns_mapping)


    CouldAssignDf_columns_mapping = {}
    # Iterate through columns in df_WReceivingVals
    for col in df_WCouldAssignVals.columns:
        # Check if the column is in groupingVarsDict
        if col not in groupingVarsDict.values():
            # If not, rename the column by appending "_CouldAssignVals"
            new_col_name = col + "_CouldAssignVals"
        else:
            # If it is, keep the original column name
            new_col_name = col
        # Add the mapping to the dictionary
        CouldAssignDf_columns_mapping[col] = new_col_name
    
    # Rename columns in df_WReceivingVals using the dictionary
    ValsCouldAssign_Df = df_WCouldAssignVals.rename(columns=CouldAssignDf_columns_mapping)


    col_ReceivingVals_Renamed = col_ReceivingVals + "_ReceivingVals"
    col_CouldAssignVals_Renamed = col_CouldAssignVals + "_CouldAssignVals"



    if groupingVarsDict:
        LeftJoined_Df = pd.merge(ReceivingVals_Df,
                                 ValsCouldAssign_Df,
                                 how = "left",
                                 left_on = list(groupingVarsDict.keys()),
                                 right_on = list(groupingVarsDict.values())
                                )
            

        
    else:
        LeftJoined_Df = product(ReceivingVals_Df[col_ReceivingVals_Renamed],
                            ValsCouldAssign_Df[col_CouldAssignVals_Renamed])     
        


#Get fuzzy scores, excluding cases where one of the two columns contains an NA.
    Conditions = [
        pd.isna(LeftJoined_Df[col_ReceivingVals_Renamed]) | pd.isna(LeftJoined_Df[col_CouldAssignVals_Renamed])
        # You can add more conditions here if needed
    ]

    Choices = [
        np.nan
    ]

    
    LeftJoined_RemovedNa_in_StringComparisionCols_Df = LeftJoined_Df.dropna(subset=[col_ReceivingVals_Renamed, col_CouldAssignVals_Renamed])
    
    FuzzyScores_Df = LeftJoined_RemovedNa_in_StringComparisionCols_Df

    
    FuzzyScores_Df.loc[:, "FuzzyScore"] = FuzzyScores_Df.apply(
        lambda row: jf_ElementWiseListPair(jf.damerau_levenshtein_distance, [row[col_ReceivingVals_Renamed]], [row[col_CouldAssignVals_Renamed]])[0],
        axis=1
    )

    FuzzyScores_Df = FuzzyScores_Df.sort_values(by=[col_ReceivingVals_Renamed, "FuzzyScore"])

    return FuzzyScores_Df



def server(input, output, session):

    FileOneInfo = reactive.Value()
    FileTwoInfo = reactive.Value()

    DfOne = reactive.Value()
    DfTwo = reactive.Value()

    DfDict = reactive.Value()

    
    @reactive.Effect
    def _():
        FileOneInfo.set(input.fileOneToUploadDir())

        FileTwoInfo.set(input.fileTwoToUploadDir())

        # Use req to ensure that the file paths are not None
        req(FileOneInfo(), FileTwoInfo())

        # Now set the reactive values
        DfOne.set(pd.read_csv(FileOneInfo()[0]["datapath"]))
        DfTwo.set(pd.read_csv(FileTwoInfo()[0]["datapath"]))

        DfDict.set({FileOneInfo()[0]["name"]: DfOne(),
                    FileTwoInfo()[0]["name"]: DfTwo()})

    @reactive.Effect
    def _():
        req(DfDict())
        ui.update_select("SelectDf_ReceivingVals", choices = list(DfDict().keys()))

    @reactive.Effect
    def _():
        req(input.SelectDf_ReceivingVals())

        DfKey_ReceivingVals = input.SelectDf_ReceivingVals()

        ui.update_select("SelectCol_ReceivingVals",
                                choices=list(DfDict()[DfKey_ReceivingVals].columns))
        


    @reactive.Effect
    def _():
        req(DfDict())
        ui.update_select("SelectDf_CouldAssignVals", choices = list(DfDict().keys()))

    @reactive.Effect
    def _():
        req(input.SelectDf_CouldAssignVals())

        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()

        ui.update_select("SelectCol_CouldAssignVals",
                         choices = list(DfDict()[DfKey_CouldAssignVals].columns))


    @reactive.Effect
    def _():
        req(input.SelectDf_CouldAssignVals())

        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()

        ui.update_select("SelectCol_CouldAssignVals",
                         choices = list(DfDict()[DfKey_CouldAssignVals].columns))
        

    @reactive.Effect
    def _():
        req(input.SelectDf_CouldAssignVals())

        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()

        ui.update_select("SelectCol_CouldAssignVals",
                         choices = list(DfDict()[DfKey_CouldAssignVals].columns))
        

    #Initialise grouping pairs dictionary reactive value
    GroupCols_InputIdDict = reactive.Value({})
    GroupingUiCreated = reactive.Value(None)

    @output
    @render.ui
    @reactive.event(input.NumberOfGroupingPairs)
    def GroupingPairs_Ui():
        req(input.SelectDf_ReceivingVals(),
            input.SelectDf_CouldAssignVals(),
            DfDict())
        
        GroupingUiCreated.set(None)
        GroupCols_InputIdDict.set({})
        # Get the number of pairs from the numeric input
        # Create n pairs of select inputs
        GroupingPairsUiContent = []
        
        GroupingUiDupeWarning = reactive.Value(False)

        DfKey_ReceivingVals = input.SelectDf_ReceivingVals()
        DfKey_ReceivingVals_WoFileExt = re.sub("\..+$", "", DfKey_ReceivingVals)

        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()
        DfKey_CouldAssignVals_WoFileExt = re.sub("\..+$", "", DfKey_CouldAssignVals)
        

        for i in range(1, input.NumberOfGroupingPairs() + 1):

            CurGroupColPair_InputUiId_Receiving = f"GroupCol{i}_Receiving_{DfKey_ReceivingVals_WoFileExt}"

            CurGroupColPair_InputUiId_CouldAssign = f"GroupCol{i}_CouldAssign_{DfKey_CouldAssignVals_WoFileExt}"
            

            UpdatedGroupColsDict = GroupCols_InputIdDict()
            UpdatedGroupColsDict[CurGroupColPair_InputUiId_Receiving] = CurGroupColPair_InputUiId_CouldAssign

            GroupCols_InputIdDict.set(UpdatedGroupColsDict)


      

            UiRow = ui.row(
                        ui.column(6,
                                  ui.input_select(CurGroupColPair_InputUiId_Receiving, 
                                                  label = f"Select column for grouping pair {i} - {DfKey_ReceivingVals_WoFileExt}", 
                                                  choices = list(DfDict()[DfKey_ReceivingVals].columns))

                        ),
                        ui.column(6,
                                    ui.input_select(CurGroupColPair_InputUiId_CouldAssign, 
                                                    label = f"Select column for grouping pair {i} - {DfKey_CouldAssignVals_WoFileExt}", 
                                                    choices = list(DfDict()[DfKey_CouldAssignVals].columns))
                        )
            )

            GroupingPairsUiContent.append(UiRow)
            
            GroupingUiCreated.set(True)

        #GroupingPairsUiContent.append(ui.output_text("UiGroupingPairsDupeWarning"))

        return GroupingPairsUiContent
    
    
    @reactive.Effect
    def _():
        req(input.SelectDf_ReceivingVals())
        req(input.SelectDf_CouldAssignVals())

        DfKey_ReceivingVals = input.SelectDf_ReceivingVals()
        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()

        ui.update_checkbox_group("OtherUniquifyingCols_ReceivingVals",
                                 choices = list(DfDict()[DfKey_ReceivingVals].columns))

        ui.update_checkbox_group("OtherUniquifyingCols_CouldAssignVals",
                                 choices = list(DfDict()[DfKey_CouldAssignVals].columns))
    
    

    @reactive.Effect
    def _():
        req(input.SelectDf_ReceivingVals())
        req(input.SelectDf_CouldAssignVals())
      
        DfKey_ReceivingVals = input.SelectDf_ReceivingVals()
        DfKey_CouldAssignVals = input.SelectDf_CouldAssignVals()

        ui.update_select("Select_LongCol_Receiving",
                        choices = list(DfDict()[DfKey_ReceivingVals].columns))
        
        ui.update_select("Select_LatCol_Receiving",
                         choices = list(DfDict()[DfKey_ReceivingVals].columns))



        ui.update_select("Select_LongCol_CouldAssign",
                        choices = list(DfDict()[DfKey_CouldAssignVals].columns))
    
        ui.update_select("Select_LatCol_CouldAssign",
                        choices = list(DfDict()[DfKey_CouldAssignVals].columns))
    
    



    GroupCols_Dict = reactive.Value(None)
    
    GroupingDupeWarning = reactive.Value(None)


    @reactive.Effect
    def _():
        req(GroupingUiCreated())

        PairOfGroupCols_List = []

        for key_id, value_id in GroupCols_InputIdDict().items():

         
            GroupCol_ReceivingSide = input[key_id]()
            GroupCol_CouldAssignSide = input[value_id]()


            PairOfGroupCols_List.append((GroupCol_ReceivingSide, GroupCol_CouldAssignSide))


        def has_duplicates(lst):
            seen = set()
            for item in lst:
                if item in seen:
                    return True
                seen.add(item)
            return False

        # Extract columns from selections
        columns_from_receiving = [pair[0] for pair in PairOfGroupCols_List]
        columns_from_could_assign = [pair[1] for pair in PairOfGroupCols_List]

        # Check for duplicates within each side
        if has_duplicates(columns_from_receiving) or has_duplicates(columns_from_could_assign):
            GroupingDupeWarning.set(True)

            GroupCols_Dict.set(None)

        else:
            GroupingDupeWarning.set(False)
            # Convert to list of tuples if no duplicates
            PairOfGroupCols_Dict = dict(PairOfGroupCols_List)
       
            GroupCols_Dict.set(PairOfGroupCols_Dict)




    @output
    @render.text
    def UiGroupingPairsDupeWarning():
        req(GroupingDupeWarning())
        if GroupingDupeWarning():
            return "Warning: The same column of the same dataset cannot be used in more than one pair."
        else:
            return "" 

    FuzzyMatchResults = reactive.Value(None)
    NextStageReady = reactive.Value(None)
    
    
    

    # This can be improved to let the table update with the thresholds, assuming this is desireable all things considered

    #The use of the fuzzyScoreLists will result in the grouping column names of the left handside dataset
    #being used, because of the left-join. So when dealing with the FuzzyMatchResults, the grouping
    #column names are the keys of the dictionary. These should be used instead, and as a list,
    #for the next steps.
    GroupCols = reactive.Value(None)
    
    
    FuzzyResult_ReceivingValsColName = reactive.Value(None)
    FuzzyResult_CouldAssignValsColName = reactive.Value(None)

    OtherUniquifyingColNames_ReceivingVals = reactive.Value([])
    OtherUniquifyingColNames_CouldAssignVals = reactive.Value([])


    @reactive.Effect
    def __():
        req(input.OtherUniquifyingCols_ReceivingVals() or input.OtherUniquifyingCols_CouldAssignVals())


        if input.OtherUniquifyingCols_ReceivingVals():

            OtherUniquifyingColNames_ReceivingVals.set([value + "_ReceivingVals" for value in input.OtherUniquifyingCols_ReceivingVals()])
            

        if input.OtherUniquifyingCols_CouldAssignVals():
            OtherUniquifyingColNames_CouldAssignVals.set([value + "_CouldAssignVals" for value in input.OtherUniquifyingCols_CouldAssignVals()])
                        

    AutoMatchResults = reactive.Value(None)


    

    
    def maskWListOfDicts(df, listOfDicts):
        dfIndex = df.index

        CombinedMask = pd.Series([False] * len(df), index = dfIndex)
        for cur_Dict in listOfDicts:
            Cur_Mask = (df[list(cur_Dict)] == pd.Series(cur_Dict)).all(axis=1)
            CombinedMask = CombinedMask | Cur_Mask
        
        return CombinedMask
    


    def ListOfUniqueDictsByKey(keys: list, listOfDicts: list, excludeKeys: list = None):
        result = []
        spottedCombinations = set()

        # Determine the strategy for filtering keys based on the provided parameters
        if keys is not None:
            keysToKeep = set(keys)
            filterStrategy = "include"
        elif excludeKeys is not None:
            excludeKeysSet = set(excludeKeys)
            filterStrategy = "exclude"
        else:
            filterStrategy = "none"

        for curDict in listOfDicts:
            # Apply the filtering strategy
            if filterStrategy == "include":
                filteredDict = {k: curDict[k] for k in keysToKeep if k in curDict}
            elif filterStrategy == "exclude":
                filteredDict = {k: v for k, v in curDict.items() if k not in excludeKeysSet}
            else:
                filteredDict = curDict.copy()

            # Use sorted items for consistent ordering in the uniqueness check
            valuesTuple = tuple(sorted(filteredDict.items()))

            if valuesTuple not in spottedCombinations:
                result.append(filteredDict)
                spottedCombinations.add(valuesTuple)

        return result



        
    
    AssignmentsMade_ListOfDicts = reactive.Value([])

    AssignmentOptsDismissed_ListOfDicts = reactive.Value([])

    ReceivingValIdsProcessed_List = reactive.Value([])

    SwitchOn = reactive.Value(False)

    @output
    @render.data_frame
    @reactive.event(input.RunProcess)
    
    def DisplayInitial():
        req(GroupCols_Dict()) #Eventually instead of this replace with greying out the run button and other relevant UI components.
        req(input.SelectDf_ReceivingVals(), input.SelectCol_ReceivingVals(),
            input.SelectDf_CouldAssignVals(), input.SelectCol_CouldAssignVals())


        #Rename the uniquifying columns before using fuzzyScoreLists

        Results = fuzzyScoreLists(
            df_WReceivingVals = DfDict()[input.SelectDf_ReceivingVals()],
            col_ReceivingVals = input.SelectCol_ReceivingVals(),
            df_WCouldAssignVals = DfDict()[input.SelectDf_CouldAssignVals()],
            col_CouldAssignVals = input.SelectCol_CouldAssignVals(),
            groupingVarsDict = GroupCols_Dict()
        )

        ExclusionThreshold = input.Threshold_ExcludeFuzzyScoresAbove()


        FuzzyResult_ReceivingValsColName.set(input.SelectCol_ReceivingVals() + "_ReceivingVals") # This reflects the fuzzymatching function.

        FuzzyResult_CouldAssignValsColName.set(input.SelectCol_CouldAssignVals() + "_CouldAssignVals") # This reflects the fuzzymatching function.
          
        GroupCols.set(list(GroupCols_Dict().keys()))
        

        Results = Results.query("FuzzyScore <= @ExclusionThreshold").copy()


        #Create handy and important new columns:
        Results.loc[:, "ReceivingVal_Uuid"] = Results[
                                               [FuzzyResult_ReceivingValsColName()] +
                                               GroupCols() +
                                               OtherUniquifyingColNames_ReceivingVals()
                                               ].astype(str).agg('_'.join, axis=1)
            

        Results.loc[:, "CouldAssignVal_Uuid"] = Results[
                                                [FuzzyResult_CouldAssignValsColName()] +
                                                 GroupCols() +
                                                 OtherUniquifyingColNames_CouldAssignVals()
                                                 ].astype(str).agg('_'.join, axis=1)

        Results.loc[:, "ValPairing_Uuid"] = Results[
                                              [FuzzyResult_ReceivingValsColName()] +
                                              OtherUniquifyingColNames_ReceivingVals() +
                                              GroupCols() +
                                              [FuzzyResult_CouldAssignValsColName()] +
                                              OtherUniquifyingColNames_CouldAssignVals()
                                           ].astype(str).agg('_'.join, axis=1)

        Results.loc[:, "Assignment_Decision"] = None



        if input.AutoMatchExactMatches():

            # Assuming 'Results' is your DataFrame and the functions return column names as strings or lists of column names

            # Step 1: Create composite columns
            Results.loc[:, 'ReceivingValWGroups'] = Results[[FuzzyResult_ReceivingValsColName()] + GroupCols()].astype(str).agg('_'.join, axis=1)
            Results.loc[:, 'CouldAssignValWGroups'] = Results[[FuzzyResult_CouldAssignValsColName()] + GroupCols()].astype(str).agg('_'.join, axis=1)

            # Step 2: Create "String exact match" column
            Results.loc[:, 'StringExactMatch'] = Results['ReceivingValWGroups'] == Results['CouldAssignValWGroups']



            
 
            # Step 4 & 5: Count distinct ValPairWUniqueId, grouped by ReceivingValWGroups and CouldAssignValWGroups
            # and create a mask for counts == 1

            grouped = Results.groupby(['ReceivingValWGroups', 'CouldAssignValWGroups'])
            Results.loc[:, 'ValPairCount'] = grouped['ValPairing_Uuid'].transform(lambda x: x.nunique())


            Mask_ExactMatchInclUniquifyingCols = (Results["StringExactMatch"] == True) & (Results['ValPairCount'] == 1)

             #Drop temporary columns (keep receiving and could-assign UUID columns, Assignment_Decision) 
            Results = Results.drop(columns=["ReceivingValWGroups", "CouldAssignValWGroups", "StringExactMatch", "ValPairCount"])

            AutoMatchResults_StaticDf = Results.copy()
            AutoMatchResults_StaticDf = AutoMatchResults_StaticDf[Mask_ExactMatchInclUniquifyingCols]
            AutoMatchResults_StaticDf["Assignment_Decision"] = "Confirmed"

            AutoMatchResults_ListOfDicts = AutoMatchResults_StaticDf[["ReceivingVal_Uuid", "CouldAssignVal_Uuid", "ValPairing_Uuid", "Assignment_Decision"]].to_dict(orient='records')



        else:
            AutoMatchResults_ListOfDicts = []

        
        
        #Obtain lists of dictionaries from the imported save file
        if input.UploadDir_ProgressSaveFile():
        
            ToImportProgressSaveFile_FilePath = input.UploadDir_ProgressSaveFile()[0]["datapath"]


            with open(ToImportProgressSaveFile_FilePath, 'r') as file:
                ImportedSave_MegaDict = json.load(file)


            AssignmentsMade_ImportedSaveListOfDicts = ImportedSave_MegaDict["AssignmentsMade"]

            AssignmentOptsDismissed_ImportedSaveListOfDicts = ImportedSave_MegaDict["AssignmentOptsDismissed"]

        else: 
            AssignmentsMade_ImportedSaveListOfDicts = []
            AssignmentOptsDismissed_ImportedSaveListOfDicts = []

        
        #Combine FuzzyMatch and imported save file lists of dictionaries, removing duplicates.
        #What this does is it allows both sources to be considered, but only upon addressing the intersection of the two.
        
        def combine_unique_lists_of_dicts(list1, list2, unique_key):

            # Combine both lists
            combined_list_of_dicts = list1 + list2

            # Use a set to track seen unique keys
            seen_unique_keys = set()

            # List to hold combined results without duplicates
            combined_unique_list_of_dicts = []

            for item in combined_list_of_dicts:
                unique_value = item[unique_key]
                if unique_value not in seen_unique_keys:
                    # If the unique value hasn't been seen, add the item to the result list
                    combined_unique_list_of_dicts.append(item)
                    seen_unique_keys.add(unique_value)

            # Now, combined_unique_list_of_dicts contains combined items without duplicates
            return combined_unique_list_of_dicts
        
        CombinedAssignmentsMade_ListOfDicts = combine_unique_lists_of_dicts(AssignmentsMade_ImportedSaveListOfDicts,
                                                                            AutoMatchResults_ListOfDicts,
                                                                            "ValPairing_Uuid")
            

        #No need to combine with another list yet, but will use the name for easy integration of the need arises.
        CombinedAssignmentOptsDismissed_ListOfDicts = AssignmentOptsDismissed_ImportedSaveListOfDicts            


        #Now extend the corresponding *reactive* ListOfDicts with the combined list of dicts
        ToUpdate_AssignmentsMade_ListOfDicts = AssignmentsMade_ListOfDicts()
        ToUpdate_AssignmentsMade_ListOfDicts.extend(CombinedAssignmentsMade_ListOfDicts)  # Append to it / extend it
        AssignmentsMade_ListOfDicts.set(ToUpdate_AssignmentsMade_ListOfDicts)
        
        
        ToUpdate_AssignmentOptsDismissed_ListOfDicts = AssignmentOptsDismissed_ListOfDicts()
        ToUpdate_AssignmentOptsDismissed_ListOfDicts.extend(CombinedAssignmentOptsDismissed_ListOfDicts)  # Append to it / extend it
        AssignmentOptsDismissed_ListOfDicts.set(ToUpdate_AssignmentOptsDismissed_ListOfDicts)

        



        FuzzyMatchResults.set(Results)

        SwitchOn.set(True)
        NextStageReady.set(True)

        return render.DataTable(FuzzyMatchResults())



    #sorted ou in UI GroupCols = reactive.Value(None)
    FuzzyMatchResults_Ordered = reactive.Value(None)

    FuzzyMatchesLeft_DuringAssignment_UpdateInLoop = reactive.Value(None)

    Df_AllUnique_ReceivingValsWGroupCols = reactive.Value(None)
    

    ListOfReceivingValIds_Initial = reactive.Value(None)

    ListOfReceivingValIds_Now = reactive.Value(None)

    
#In principle this should probably absorb the above. Keeping for now for consideration of options.
    @reactive.Effect
    def _():
        req(NextStageReady())

        print("Block 1")
        with reactive.isolate():

            # Sort the DataFrame
            sorted_df = FuzzyMatchResults().sort_values(by=["FuzzyScore"] +
                                                           [FuzzyResult_ReceivingValsColName()] +
                                                        GroupCols() +
                                                        OtherUniquifyingColNames_ReceivingVals())

            # Define the desired column order
            desired_column_order = [FuzzyResult_ReceivingValsColName(),
                                    FuzzyResult_CouldAssignValsColName(),
                                    "FuzzyScore",
                                    "CouldAssignVal_Uuid"] + GroupCols()

            # Get the list of columns not mentioned in the desired order
            remaining_columns = [col for col in sorted_df.columns if col not in desired_column_order]

            # Concatenate the desired order with the remaining columns
            new_column_order = desired_column_order + remaining_columns

            # Reindex the DataFrame with the new column order
            reordered_df = sorted_df.reindex(columns = new_column_order)


            FuzzyMatchResults_Ordered.set(reordered_df)


            Df_AllUnique_ReceivingValsWGroupCols.set(FuzzyMatchResults_Ordered()[GroupCols() +
                                                                                OtherUniquifyingColNames_ReceivingVals() +
                                                                                [FuzzyResult_ReceivingValsColName()] +
                                                                                ["ReceivingVal_Uuid"]]
                                                    .drop_duplicates()
                                                 )
            


            
            ListOfReceivingValIds_Initial.set(

                Df_AllUnique_ReceivingValsWGroupCols()['ReceivingVal_Uuid'].tolist()
                
            )



    #Run only once (unless restarting whole process)
            
    ListOfReceivingValIds_Rearranged = reactive.Value(None)

    
    @reactive.Effect
    def __():
        req(SwitchOn())
        req(ListOfReceivingValIds_Initial())

        ListOfReceivingValIds_Rearranged.set(

            ListOfReceivingValIds_Initial()
                
            )
        
        SwitchOn.set(False)



    @reactive.Effect
    def __():
        req(ListOfReceivingValIds_Rearranged())

        print("Block 1.5")
        
        TriggerNextCycle()

        AssignedReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentsMade_ListOfDicts())
            
        DismissedReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentOptsDismissed_ListOfDicts())


        with reactive.isolate():

            ReceivingValIdsCycledThrough_List = {d['ReceivingVal_Uuid'] for d in AssignedReceivingVals_ListOfDicts + DismissedReceivingVals_ListOfDicts}

            ListOfReceivingValIds_Rearranged_Static = ListOfReceivingValIds_Rearranged()

            ListOfReceivingValIds_Now_Static = [receivingValId for receivingValId in ListOfReceivingValIds_Rearranged_Static if receivingValId not in ReceivingValIdsCycledThrough_List]
            
            ListOfReceivingValIds_Now.set(ListOfReceivingValIds_Now_Static)
            
    
    @output
    @render.text
    def CurrentStageTextFeedback():
        req(ListOfReceivingValIds_Now())

        print("Block 2")


        CurCycle_ReceivingValNum = (len(ListOfReceivingValIds_Initial()) - len(ListOfReceivingValIds_Now())) + 1

        return f"Receiving value {CurCycle_ReceivingValNum} of {len(ListOfReceivingValIds_Initial())}"

        
    CurReceivingValId = reactive.Value(None)

    AssignmentsMade_Df = reactive.Value(None)
    
    DismissedOpts_Df = reactive.Value(None)
    

    
    DfDisplay_Cur_FuzzyMatchesUnderConsideration = reactive.Value(None)
    
    @reactive.Effect
    def _():

        req(ListOfReceivingValIds_Now())

        
        print("Block 3")
        
        Assigned_CouldAssignVals_ListOfDicts = ListOfUniqueDictsByKey(["CouldAssignVal_Uuid"], AssignmentsMade_ListOfDicts())

        print("loop check")
        Assigned_ReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentsMade_ListOfDicts())

        Dismissed_ReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentOptsDismissed_ListOfDicts())

        with reactive.isolate():

            print("Big mega block?")
            ###Filtering to create FuzzyMatchesLeft_DuringAssignment_UpdateInLoop####

            Mask_AssignedCouldAssignVals = maskWListOfDicts(FuzzyMatchResults_Ordered(), Assigned_CouldAssignVals_ListOfDicts)
            Mask_AssignedReceivingVals = maskWListOfDicts(FuzzyMatchResults_Ordered(), Assigned_ReceivingVals_ListOfDicts)
            Mask_DismissedReceivingVals = maskWListOfDicts(FuzzyMatchResults_Ordered(), Dismissed_ReceivingVals_ListOfDicts)
            
            FuzzyMatchesLeft_Static_Df = FuzzyMatchResults_Ordered()
            FuzzyMatchesLeft_Static_Df = FuzzyMatchesLeft_Static_Df[~(Mask_AssignedReceivingVals |
                                                                      Mask_DismissedReceivingVals |
                                                                      Mask_AssignedCouldAssignVals)]


            FuzzyMatchesLeft_DuringAssignment_UpdateInLoop.set(FuzzyMatchesLeft_Static_Df)

            
            print("How fast?1")
            ###Filtering to create AssignmentsMade_Df_Static, then adding the decision made to create AssignmentsMade_Df (reactive)####
            #Remove Decision_Made from dictionary for the filtering stage
            AssignmentsMade_ListOfDicts_NoDecisionCol = ListOfUniqueDictsByKey(keys = None,
                                                                            listOfDicts = AssignmentsMade_ListOfDicts(),
                                                                            excludeKeys = ["Assignment_Decision"])
                
            Mask_AssignmentsMade = maskWListOfDicts(FuzzyMatchResults_Ordered(), AssignmentsMade_ListOfDicts_NoDecisionCol)
            
        
            AssignmentsMade_Df_Static = FuzzyMatchResults_Ordered()[Mask_AssignmentsMade]

            AssignmentsMade_Df_DroppedDecisionCol = AssignmentsMade_Df_Static.drop(columns=["Assignment_Decision"])
            
            ValPairId_w_DecisionsMade_Assignments_ListOfDicts = ListOfUniqueDictsByKey(["ValPairing_Uuid", "Assignment_Decision"], AssignmentsMade_ListOfDicts())

            ValPairId_w_DecisionsMade_Assignments_Df = pd.DataFrame(ValPairId_w_DecisionsMade_Assignments_ListOfDicts)

            if ValPairId_w_DecisionsMade_Assignments_Df is not None and not ValPairId_w_DecisionsMade_Assignments_Df.empty:

                AssignmentsMade_Df_DecisionAdded = pd.merge(AssignmentsMade_Df_DroppedDecisionCol, ValPairId_w_DecisionsMade_Assignments_Df,
                                                        on="ValPairing_Uuid", how="left")
                
            else: AssignmentsMade_Df_DecisionAdded = None                

            AssignmentsMade_Df.set(AssignmentsMade_Df_DecisionAdded)

            print("How fast?2")

            ###Filtering to create DismissedOpts_Df_Static, then adding the decision made to create DismissedOpts_Df (reactive)####
            #Remove assignment decision column for the filtering part

            
            AssignmentOptsDismissed_ListOfDicts_NoDecisionCol = ListOfUniqueDictsByKey(keys = None,
                                                                            listOfDicts = AssignmentOptsDismissed_ListOfDicts(),
                                                                            excludeKeys = ["Assignment_Decision"])
            
            Mask_DismissedOpts = maskWListOfDicts(FuzzyMatchResults_Ordered(), AssignmentOptsDismissed_ListOfDicts_NoDecisionCol)
            
            DismissedOpts_Df_Static = FuzzyMatchResults_Ordered()[Mask_DismissedOpts]

            DismissedOpts_Df_DroppedDecisionCol = DismissedOpts_Df_Static.drop(columns=["Assignment_Decision"])
    
            ValPairId_w_DecisionsMade_Dismissed_ListOfDicts = ListOfUniqueDictsByKey(["ValPairing_Uuid", "Assignment_Decision"], AssignmentOptsDismissed_ListOfDicts())

            ValPairId_w_DecisionsMade_Dismissed_Df = pd.DataFrame(ValPairId_w_DecisionsMade_Dismissed_ListOfDicts)

            if ValPairId_w_DecisionsMade_Dismissed_Df is not None and not ValPairId_w_DecisionsMade_Dismissed_Df.empty:
      
                DismissedOpts_Df_DecisionAdded = pd.merge(DismissedOpts_Df_DroppedDecisionCol, ValPairId_w_DecisionsMade_Dismissed_Df,
                                                          on = "ValPairing_Uuid", how='left')
                
            else: DismissedOpts_Df_DecisionAdded = None

            DismissedOpts_Df.set(DismissedOpts_Df_DecisionAdded)

            ###Create the dataframe that will be used for ResponsiveDf
            ###(this will be how this entire block is triggered, as ListOfReceivingValIds_Now will be updated by the decision button processes)####
            print("How fast?2.5")

            Assigned_ReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentsMade_ListOfDicts())

            Dismissed_ReceivingVals_ListOfDicts = ListOfUniqueDictsByKey(["ReceivingVal_Uuid"], AssignmentOptsDismissed_ListOfDicts())

            print("How fast?3")

            CurReceivingValId.set(ListOfReceivingValIds_Now()[0])

            CurReceivingValId_Static = CurReceivingValId() 

            DfDisplay_Cur_FuzzyMatchesUnderConsideration.set(FuzzyMatchesLeft_DuringAssignment_UpdateInLoop().query("ReceivingVal_Uuid == @CurReceivingValId_Static"))                  
            
            print("How fast?4")



        

    def create_query_from_dict(dict_obj):
            return " & ".join([f"`{k}` == \"{v}\"" if "'" in v else f"`{k}` == '{v}'" for k, v in dict_obj.items()])
            


                
    @output
    @render.data_frame
    def ResponsiveDisplay():

        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
             not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        
        print("Block 4")

        return render.DataGrid(DfDisplay_Cur_FuzzyMatchesUnderConsideration(), row_selection_mode = "single")
        
    
        
    LongCol_Receiving = reactive.Value()
    LatCol_Receiving = reactive.Value()
    LongCol_CouldAssign = reactive.Value()
    LatCol_CouldAssign = reactive.Value()

    @reactive.Effect
    def __():
        req(input.Select_LatCol_Receiving(),
            input.Select_LongCol_Receiving(),
            input.Select_LatCol_CouldAssign(),
            input.Select_LongCol_CouldAssign())
          # Get the selected column names from the input
        LatCol_Receiving.set(input.Select_LatCol_Receiving() + "_ReceivingVals")
        LongCol_Receiving.set(input.Select_LongCol_Receiving() + "_ReceivingVals")
        LatCol_CouldAssign.set(input.Select_LatCol_CouldAssign() + "_CouldAssignVals")
        LongCol_CouldAssign.set(input.Select_LongCol_CouldAssign() + "_CouldAssignVals")


    MapObj = reactive.Value()

    DataGrid_Map_Bridge = reactive.Value()

    ShpAsGeoJson = reactive.Value()


    @render_widget
    def AssignmentOptsMap():

        print("Block 5")
        MapTileLayer = TileLayer(
            url="http://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            )

        MapObj.set(Map(layers=(MapTileLayer,),
                       center=(2.416769, 42.96891), 
                       zoom = 9)
                 )
        
        
        DrawControl_Obj = DrawControl(circlemarker={})

        DrawControl_Obj.polyline = {
        "shapeOptions": {
        "color": "#6bc2e5",
        "weight": 8,
        "opacity": 0.8
         }
        }

        MapObj().add_control(DrawControl_Obj)

        
        return MapObj()
    

    @reactive.Effect
    def __():
        req(input.ShapeFileToUploadDir())

        print("Block 6")

        ShapeFileFolderPath = input.ShapeFileToUploadDir()

        ListOfFiles = os.listdir(ShapeFileFolderPath)
        
        ui.update_select("SelectShapeFile", choices = ListOfFiles)
        
        
    ShapeFileUsed = reactive.Value()

    @reactive.Effect
    def __():

        print("Block 7.1")
        ShapeFileName = input.SelectShapeFile()

        # Check if ShapeFileName is not None and ends with .shp
        if ShapeFileName and re.search(r"\.shp$", ShapeFileName):
            ShapeFileUsed.set(True)
        else: 
            ShapeFileUsed.set(False)

        # Proceed only if a shapefile is selected
        req(ShapeFileUsed())

        print("Block 7.2")


        ShapeFilePath = input.ShapeFileToUploadDir() + "\\" + ShapeFileName
 
        ShapeFile = gpd.read_file(ShapeFilePath)

        # Convert the GeoDataFrame to GeoJSON
        ShpAsGeoJson_Static = json.loads(ShapeFile.to_json())
            
        # Update the reactive value with the GeoJSON data
        ShpAsGeoJson.set(ShpAsGeoJson_Static)
            
        # Add the GeoJSON layer to the map
        MapObj().add_layer(GeoJSON(data = ShpAsGeoJson_Static))

        
    SelectedRowIndices = reactive.Value(None)

    #Create a reactive to hold the input values from the responsive datatable (so it can be reset before the next cycle)
    @reactive.Effect
    def _():
        req(input.ResponsiveDisplay_selected_rows())

        print("Block 8")

        SelectedRowIndices.set(input.ResponsiveDisplay_selected_rows())

    
    @reactive.Effect
    def update_map():
        # Ensure that the DataFrame and selection inputs are not empty
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
            not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        
        
        if SelectedRowIndices():  # Check if any row is selected
           SelectedRowIndex = SelectedRowIndices()[0]

           print("Block 9")

                
           Selected_CouldAssignUuid = DfDisplay_Cur_FuzzyMatchesUnderConsideration().iloc[SelectedRowIndex]["CouldAssignVal_Uuid"]
           DataGrid_Map_Bridge.set(Selected_CouldAssignUuid)

        else:
            DataGrid_Map_Bridge.set(None)


        df = DfDisplay_Cur_FuzzyMatchesUnderConsideration()
          
                
        if DataGrid_Map_Bridge(): 
            Mask_SelectedRow = df["CouldAssignVal_Uuid"] == DataGrid_Map_Bridge()
                

            Df_SelectedCouldAssign = df[Mask_SelectedRow]


            df = df[~Mask_SelectedRow]


        if ShapeFileUsed():
            N_NonPointLayers = 2
        else:
            N_NonPointLayers = 1


        if len(MapObj().layers) > N_NonPointLayers:
            # Collect layers to remove (skipping the base layer at index 0)
            layers_to_remove = MapObj().layers[N_NonPointLayers:]
            for layer in layers_to_remove:
                MapObj().remove(layer)

        #If statement necessary to deal with dataframes that are already single-rowed.
        if len(df) > 0:
            MapObj().center = (df.iloc[0][LatCol_Receiving()], df.iloc[0][LongCol_Receiving()])
        
        elif len(df) == 0:
            MapObj().center = (Df_SelectedCouldAssign.iloc[0][LatCol_Receiving()], Df_SelectedCouldAssign.iloc[0][LongCol_Receiving()])


        receiving_layer_group = LayerGroup(name = "Receiving")
        could_assign_layer_group = LayerGroup(name = "CouldAssign")
        could_assign_selected_layer_group = LayerGroup(name = "SelectedCouldAssign")

        # Iterate through the DataFrame to add markers to their respective LayerGroups

        # Initialize an empty list to hold the markers for "CouldAssign" points
        could_assign_markers = []

        if len(df) > 0:
            for index, row in df.iterrows():
                # Check for non-null lat/long and create a marker
                if pd.notnull(row[LatCol_CouldAssign()]) and pd.notnull(row[LongCol_CouldAssign()]):
                    marker = Marker(location=(row[LatCol_CouldAssign()], row[LongCol_CouldAssign()]),
                                    title=row['CouldAssignVal_Uuid'],
                                    draggable=False)
                    could_assign_markers.append(marker)

            # Create a MarkerCluster from the list of markers
            could_assign_cluster = MarkerCluster(markers=could_assign_markers,
                                                radius=7, 
                                                disableClusteringAtZoom = 10,
                                                spiderfyOnMaxZoom = None)

            # Add the MarkerCluster to the map instead of a LayerGroup
            MapObj().add_layer(could_assign_cluster)


        
        if len(df) > 0:   
            ReceivingValMarker = CircleMarker(location = (df.iloc[0][LatCol_Receiving()], df.iloc[0][LongCol_Receiving()]),
                                            radius=9, 
                                            color='green',
                                            fill_color='white',
                                            fill_opacity=0.5, 
                                            opacity=0.8)
            
        elif len(df) == 0:
            ReceivingValMarker = CircleMarker(location = (Df_SelectedCouldAssign.iloc[0][LatCol_Receiving()], Df_SelectedCouldAssign.iloc[0][LongCol_Receiving()]),
                                             radius=9, 
                                             color='green',
                                             fill_color='white',
                                             fill_opacity=0.5, 
                                             opacity=0.8)
            
            
        receiving_layer_group.add_layer(ReceivingValMarker)

 
        if DataGrid_Map_Bridge():            
                         
            SelectedMarker =  CircleMarker(location = (Df_SelectedCouldAssign.iloc[0][LatCol_CouldAssign()],
                                                       Df_SelectedCouldAssign.iloc[0][LongCol_CouldAssign()]),
                                           radius=9, 
                                           color='red',
                                           fill_color='white',
                                           fill_opacity=0.5, 
                                           opacity=0.8)
                    
            could_assign_selected_layer_group.add_layer(SelectedMarker)
            
        # Add the LayerGroups to the Map
        MapObj().add(receiving_layer_group)
        MapObj().add(could_assign_selected_layer_group)

        print("Block 9 end")
    


    DecisionMade = reactive.Value()

    @reactive.Effect
    @reactive.event(input.ConfirmButton)
    def __():
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
                    not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty and SelectedRowIndices())
        


        print("Block 10")
                
        if SelectedRowIndices():  # Check if any row is selected
            DecisionMade.set(None)
            DecisionMade.set("Confirm") 



    @reactive.Effect
    @reactive.event(input.MarkAsUncertainGuessButton)
    def __():
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
            not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        
        print("Block 11")
        DecisionMade.set(None)
        DecisionMade.set("Uncertain guess")


    @reactive.Effect
    @reactive.event(input.FlagButton)
    def __():
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
            not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        
        print("Block 12")

        DecisionMade.set(None)
        DecisionMade.set("Flag")


    @reactive.Effect
    @reactive.event(input.DismissButton)
    def __():
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
                not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        
        print("Block 13")    
        DecisionMade.set(None)
        DecisionMade.set("Dismiss")


    @reactive.Effect
    @reactive.event(input.SkipButton)
    def __():
        req(DfDisplay_Cur_FuzzyMatchesUnderConsideration() is not None and
            not DfDisplay_Cur_FuzzyMatchesUnderConsideration().empty)
        print("Block 13--")                 
        DecisionMade.set(None)
        DecisionMade.set("Skip")  


    TriggerNextCycle = reactive.Value(None)


    @reactive.Effect
    @reactive.event(DecisionMade)
    def confirmBlock():

        req(DecisionMade() == "Confirm" or DecisionMade() == "Uncertain guess")
        
        print("Block 14")


        with reactive.isolate():

            if SelectedRowIndices():  # Check if any row is selected

                print("Block 14.2")

                SelectedRowIndex = SelectedRowIndices()[0]

                Selected_CouldAssignVals_Row = DfDisplay_Cur_FuzzyMatchesUnderConsideration().iloc[SelectedRowIndex]

                Selected_ValPairUuid_Str = Selected_CouldAssignVals_Row["ValPairing_Uuid"]

                Selected_ReceivingVal_Str = Selected_CouldAssignVals_Row["ReceivingVal_Uuid"]

                Selected_CouldAssignVal_Str = Selected_CouldAssignVals_Row["CouldAssignVal_Uuid"]
                
            
            req(SelectedRowIndices())
                
            print("Block 14.3")

            AssignmentDecisionStr = "Confirmed" if DecisionMade() == "Confirm" else "Uncertain guess" if DecisionMade() == "Uncertain guess" else None

            ToUpdateAssignmentsMade_ListOfDicts = AssignmentsMade_ListOfDicts()


            ToUpdateAssignmentsMade_ListOfDicts.append({"ReceivingVal_Uuid": Selected_ReceivingVal_Str,
                                                                                              "CouldAssignVal_Uuid": Selected_CouldAssignVal_Str,
                                                                                              "ValPairing_Uuid": Selected_ValPairUuid_Str,
                                                                                              "Assignment_Decision": AssignmentDecisionStr
                                                                                            })
                                                    
            AssignmentsMade_ListOfDicts.set(ToUpdateAssignmentsMade_ListOfDicts)


            #This should also re-trigger the block further above that applies the lists of dictionaries
            #Doing so should also render the dataframe



            TriggerNextCycle.set(None)
        
        
        TriggerNextCycle.set(True)
        SelectedRowIndices.set(None)

                
    
    Df_ValPair_AntiAssigned = reactive.Value(pd.DataFrame())
    
    @reactive.Effect
    @reactive.event(DecisionMade)
    def antiAssignedBlock(): 
        req(DecisionMade() == "Flag" or DecisionMade() == "Dismiss")

        print("Block 15")
        with reactive.isolate():

            AssignmentDecisionStr = "Pairing options flagged" if DecisionMade() == "Flag" else "There is no correct pairing" if DecisionMade() == "Dismiss" else None
            

            Df_Cur_FuzzyMatchesUnderConsideration_WDecision = DfDisplay_Cur_FuzzyMatchesUnderConsideration().copy()

            Df_Cur_FuzzyMatchesUnderConsideration_WDecision["Assignment_Decision"] = AssignmentDecisionStr
            
            NewDismissedOpts_ListOfDicts = (Df_Cur_FuzzyMatchesUnderConsideration_WDecision[["ValPairing_Uuid", "ReceivingVal_Uuid", "CouldAssignVal_Uuid", "Assignment_Decision"]]).to_dict(orient='records')

      
      
            
            AssignmentOptsDismissed_ListOfDicts_Static = AssignmentOptsDismissed_ListOfDicts()

            AssignmentOptsDismissed_ListOfDicts_Static.extend(NewDismissedOpts_ListOfDicts)
            AssignmentOptsDismissed_ListOfDicts.set(AssignmentOptsDismissed_ListOfDicts_Static)   


            TriggerNextCycle.set(None)

        TriggerNextCycle.set(True)
        SelectedRowIndices.set(None)

    SkippedReceivingValId = reactive.Value(None)

    @reactive.Effect
    @reactive.event(DecisionMade)
    def skipBlock():  
        req(DecisionMade() == "Skip")

        print("Block 16")
        with reactive.isolate():

            # Idea is to make it by updating this list here, then the application of the filters can still be done
            # else where in each cycle, but ListOfReceivingValIds_Now() is permanently changed in terms of its
            # order.
            
            ListOfReceivingValIds_Rearranged_Static = ListOfReceivingValIds_Rearranged()

            Hold_ListOfReceivingValIds_Rearranged = ListOfReceivingValIds_Rearranged_Static

            Hold_ListOfReceivingValIds_Rearranged.remove(CurReceivingValId())

            MoveToBack_ListOfReceivingValIds_Rearranged = [CurReceivingValId()]

            ListOfReceivingValIds_Rearranged_NewlyRearranged = Hold_ListOfReceivingValIds_Rearranged
            ListOfReceivingValIds_Rearranged_NewlyRearranged.extend(MoveToBack_ListOfReceivingValIds_Rearranged)

            ListOfReceivingValIds_Rearranged.set(ListOfReceivingValIds_Rearranged_NewlyRearranged)

            TriggerNextCycle.set(None)

        TriggerNextCycle.set(True)
        SelectedRowIndices.set(None)                    
        
    

    @output
    @render.data_frame
    def Display_ValPairAssignmentsMade():
        req(AssignmentsMade_Df() is not None and not AssignmentsMade_Df().empty)

        return render.DataTable(AssignmentsMade_Df())


    @output
    @render.data_frame
    def Display_ValPair_AntiAssignments():
        req(DismissedOpts_Df() is not None and not DismissedOpts_Df().empty)
        return render.DataTable(DismissedOpts_Df())
    
    @output
    @render.data_frame
    def Display_ValPairAssignmentsRemaining():

        #will probbaly need an if statement for when it becomes empty
        req(FuzzyMatchesLeft_DuringAssignment_UpdateInLoop() is not None and not FuzzyMatchesLeft_DuringAssignment_UpdateInLoop().empty)
        return render.DataTable(FuzzyMatchesLeft_DuringAssignment_UpdateInLoop())
    
    

    @session.download(filename = "Value pairs.csv")
    def ExportAssignedMatches():
        yield AssignmentsMade_Df().to_csv(index = False)

         
    @session.download(filename = "Anti_assigned.csv")
    def ExportAntiAssignedMatches():
        yield DismissedOpts_Df().to_csv(index = False)


    @session.download(filename = "Unassigned options.csv")
    def ExportRemaining():
        yield FuzzyMatchesLeft_DuringAssignment_UpdateInLoop().to_csv(index = False)


    ProgressSaveJsonStr = reactive.Value()

    @reactive.Effect
    def _():
        req(CurReceivingValId())

        print("Saving progress block")

        SaveProgressGrandDict = {
            "AssignmentsMade": AssignmentsMade_ListOfDicts(),

            "AssignmentOptsDismissed": AssignmentOptsDismissed_ListOfDicts()
            }
        
        ProgressSaveJsonStr.set(json.dumps(SaveProgressGrandDict, indent=4))

    
    @render.download(
        filename=f'ProgressSave_{time.strftime("%Y%m%d-%H%M%S")}.json'
    )

    async def ExportProgressSaveFile():
        await asyncio.sleep(0.25)
        yield ProgressSaveJsonStr()
  
# next step kinda note to self, if satisfied with number, select, then


app = App(app_ui, server, debug = False)
