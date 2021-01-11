import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    import pandas as pd


# The library path is where to find the default input data for LandBOSSE.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if ROOT.endswith("wisdem"):
    library_path = os.path.join(ROOT, "library", "landbosse")
else:
    library_path = os.path.join(ROOT, "project_input_template", "project_data")


class OpenMDAODataframeCache:
    """
    This class does not need to be instantiated. This means that the
    cache is shared throughout all parts of the code that needs access
    to any part of the project_data .xlsx files.

    This class is made to read all sheets from xlsx files and store those
    sheets as dictionaries. This is so .xlsx files only need to be parsed
    once.

    One of the use cases for this dataframe cache is in parallel process
    execution using ProcessPoolExecutor. Alternatively, once code use
    the ThreadPoolExecutor (though that wouldn't give the same advantages
    of paralelization).

    Regardless of which executor is used, care must be taken that one thread
    or process cannot mutate the dataframes of another process. So, this
    class make copies of dataframes so the callables running from the
    executor cannot overwrite each other's data.
    """

    # _cache is a class attribute that holds the cache of sheets and their
    # dataframes
    _cache = {}

    @classmethod
    def read_all_sheets_from_xlsx(cls, xlsx_basename, xlsx_path=None):
        """
        If the .xlsx file specified by .xlsx_basename has been read before
        (meaning it is stored as a key on cls._cache), a copy of all the
        dataframes stored under that sheet name is returned. See the note
        about copying in the class docstring for why copies are being made.

        If the xlsx_basename has not been read before, all the sheets are
        read and copies are returned. The sheets are stored on the dictionary
        cache.

        Parameters
        ----------
        xlsx_basename : str
            The base name of the xlsx file to read. This name should
            not include the .xlsx at the end of the filename. This class
            uses XlsxFileOperations to find the dataframes in the
            project_data directory. The xlsx_basename becomes the key
            in the dictionary used to access all the sheets in the
            named .xlsx file.

        xlsx_path : str
            The path from which to read the .xlsx file. This parameter
            has the default value of the library path variable above.

        Returns
        -------
        dict
            A dictionary of dataframes. Keys on the dictionary are names of
            sheets and values in the dictionary are dataframes in that
            .xlsx file.
        """
        if xlsx_basename in cls._cache:
            original = cls._cache[xlsx_basename]
            return cls.copy_dataframes(original)

        if xlsx_path is None:
            xlsx_filename = os.path.join(library_path, f"{xlsx_basename}.xlsx")
        else:
            xlsx_filename = os.path.join(xlsx_path, f"{xlsx_basename}.xlsx")

        xlsx = pd.ExcelFile(xlsx_filename, engine="openpyxl")
        sheets_dict = {sheet_name: xlsx.parse(sheet_name) for sheet_name in xlsx.sheet_names}
        for sheet_name in xlsx.sheet_names:
            sheets_dict[sheet_name].dropna(inplace=True, how="all")
        cls._cache[xlsx_basename] = sheets_dict
        return cls.copy_dataframes(sheets_dict)

    @classmethod
    def copy_dataframes(cls, dict_of_dataframes):
        """
        This copies a dictionary of dataframes. See the class docstring for an
        explanation of why this copying is taking place.

        Parameters
        ----------
        dict_of_dataframes : dict
            The dictionary of dataframes to copy.

        Returns
        -------
        dict
            Keys are the same as the original dictionary of dataframes.
            Values are copies of the origin dataframes.
        """
        return {xlsx_basename: df.copy() for xlsx_basename, df in dict_of_dataframes.items()}
