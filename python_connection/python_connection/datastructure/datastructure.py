import arff
import pandas as pd

class PandasDataFrameWrapper:
    def __init__(self, arff_data):
        self.X, self.y = create_pandas_dataframe(arff_data)

def create_pandas_dataframe(arff_parsed):
    dataset = list()
    y = list()
    for instance_id,instance in enumerate(arff_parsed['data']):
        for attribute_id, attribute_value in enumerate(instance):
            #no the target, but a sensor series
            if isinstance(attribute_value, str):
                split_attribute_value = attribute_value.split()
                for time_step_value_pair in split_attribute_value:
                    if not '#' in time_step_value_pair:
                        print("INVALID FORMAT!") #TODO raise exception
                    else:
                        split_pair = time_step_value_pair.split('#')
                        timestep = pd.to_numeric(split_pair[0]) #TODO robustness
                        value = pd.to_numeric(split_pair[1]) #TODO robustness
                        row = [instance_id, timestep, str(arff_parsed['attributes'][attribute_id][0]), value]
                        dataset.append(row)
            # target
            else:
                y.append(pd.to_numeric(attribute_value)) #TODO robustness
    return pd.DataFrame(dataset, columns=['instance_id','timestep','sensor','value']).astype({'instance_id': 'int32','timestep': 'int32', 'sensor': 'string', 'value': 'float32'}), pd.Series(y)


def parse(arff_, as_rul=False, is_path=True, dense_mode=True):
    """ Opens and reads the file located at path.
    May also be called with the content string.
    arff_: either  arff file path or arff file content. treated based on how is_bath is assigned.
    is_path: bool, if true, arff_ is an arff-file path. If false, arff_ is treated as the content of an arff file
    as a string.
    Returns an ArffStructure object or None if there was an error.
    """
    if is_path:
        arff_data = open(arff_, 'r')
        if as_rul:
            arff_data = arff_data.read().replace("timeseries", "string") #TODO ROBUSTNESS
    else:
        arff_data = arff_  # here path is actually the content of an arff file.
    try:
        try:
            if dense_mode:
                mode = arff.DENSE
            else:
                mode = arff.LOD
            arff_parsed = arff.load(arff_data, return_type=mode, encode_nominal=True)
        except:  # exception is thrown when sparse data is loaded in DENSE mode.
            if dense_mode:
                arff_parsed = parse(arff_, as_rul, is_path, False)  # arff may be in sparse format.

        if not as_rul:
            obj = ArffStructure(arff_parsed)
        else:
            obj = PandasDataFrameWrapper(arff_parsed)
        return obj
    except Exception as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        return None  # return None to signify an error. (Raising Exception might be a better solution)
    finally:
        if is_path and not as_rul:  # close file if necessary
            arff_data.close()


def parse_raw(arff_, is_path=True, dense_mode=True):
    """ Opens and reads the file located at path.
    May also be called with the content string.
    arff_: either  arff file path or arff file content. treated based on how is_bath is assigned.
    is_path: bool, if true, arff_ is an arff-file path. If false, arff_ is treated as the content of an arff file
    as a string.
    Returns an ArffStructure object or None if there was an error.
    """
    if is_path:
        arff_data = open(arff_, 'r')
        arff_data = arff_data.read().replace("timeseries", "string").replace("TIMESERIES", "string") #TODO ROBUSTNESS
    else:
        arff_data = arff_  # here path is actually the content of an arff file.
    try:
        arff_parsed = None
        try:
            if dense_mode:
                mode = arff.DENSE
            else:
                mode = arff.LOD
            arff_parsed = arff.load(arff_data, return_type=mode, encode_nominal=True)
        except:  # exception is thrown when sparse data is loaded in DENSE mode.
            if dense_mode:
                arff_parsed = parse_raw(arff_, is_path, False)  # arff may be in sparse format.

        return arff_parsed
    except Exception as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        return None  # return None to signify an error. (Raising Exception might be a better solution)


class ArffStructure:
    """ Stores the arff data in a way it can be used by tensorflow.
    Args:
        arff_data: arff_data which is a dictionary returned by the arff.py module
    Instance Attributes:
        class_list: list of all the classes in this arff data
        in_size: size of the input layer of the neural network
        out_size: size of the output layer of the neural network
        entry_size: number of training entries. Its the height of the input matrix.
        input_matrix: a list of a list.  input_matrix[i][j] stores the input values of attribute j of entry i
        from the arff file.
        output_matrix: a list of a list.  output_vector[i] stores the training output of entry i
        from the arff file in a one-hot manner.
    """

    def __init__(self, arff_data):
        """ Reads and encapsulates arff_data which is a dictionary returned by the arff.py module.
        """
        # attribute list containing a tuples (x,y).
        #  x = attribute name. y = attribute type.
        attributes_list = arff_data['attributes']
        # list of data entries. data entries are lists.
        data_list = arff_data['data']

        # looking for the class attribute in the list and extracting it.
        class_index = 0  # save at which index the class attribute was found

        for tupel in attributes_list:
            if tupel[0] == 'class':
                classtupel = tupel
                self.class_list = classtupel[1]
                attributes_list.remove(tupel)  # Now attribute list only consists of input attributes
                break
            class_index += 1

        if not hasattr(self, 'class_list'):
            # class list couldn't be found. use last tuple in attributes_list instead
            self.class_list = attributes_list.pop()[1]
            class_index = len(attributes_list)

        self.in_size = len(attributes_list)
        if type(self.class_list) == str:
            # class is only a single type
            self.out_size = 1
        else:
            self.out_size = len(self.class_list)

        # Now, extract data from data_list which was taken from arff
        self.input_matrix = []
        self.output_matrix = []

        # true, if arff was in a sparse format. (Im new to python and don't know how to use duck typing here.)
        sparse_format = type(data_list[0]) == dict

        # stores the accumulated values for every entry in attribute.
        # We need this to fill missing data with the median of attribute values. *See end of method*
        acc_list = [0] * (self.in_size + 1)
        missing_list = []  # list of tuple. stores the position of every missing entry.
        entry_index = 0
        for entry in data_list:
            self.input_matrix.append([])  # append a new empty row
            # every entry contains a value for each attribute. Extract it and add it to the value_list
            # for every attribute extract value
            for attr_index in range(self.in_size + 1):
                if sparse_format:
                    # If sparse_format, entry is a dict (x,y). x is index of attribute. y is value.
                    if attr_index in entry:  # value wasn't omitted.
                        value = (entry[attr_index])  # may return None if "?" value was used
                    else:  # was omitted in the arff data. So the value is 0.
                        value = 0
                else:
                    # Not sparse_format so entry is a list of every value.
                    value = entry[attr_index]  # may return None if "?" value was used
                # value has been extracted. Now decide where to store it
                if attr_index == class_index:  # value is class number. add to output
                    # one hot the output to be used with soft max in Tensorflow
                    self.output_matrix.append([])
                    for class_index_ in range(self.out_size):
                        self.output_matrix[-1].append(1 if class_index_ == value else 0)
                else:  # add to input
                    self.input_matrix[-1].append(value)  # add value to the last row
                    if value is not None:  # value is available
                        acc_list[attr_index] += value
                    else:  # value is missing. add entry to missing_list
                        missing_list.append((entry_index, attr_index))

                    acc_list[attr_index] += value if value else 0  # accumulate value for this attribute

            entry_index += 1
        # store data set length
        self.entry_size = len(self.input_matrix)

        # substitute missing values with the average of the attribute
        for tupel in missing_list:
            self.input_matrix[tupel[0]][tupel[1]] = acc_list[tupel[1]] / len(data_list)  # mean of this attribute