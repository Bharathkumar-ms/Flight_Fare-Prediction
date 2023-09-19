from src.components.data_ingestion import *
from src.components.data_cleaning import *
from src.components.data_transformation import *
from src.components.model_trainer import *
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data_imgestion=obj.initiate_data_ingestion()

    obj2=DataCleaning()
    raw_arr=obj2.initiate_data_cleaning()

    obj3=DataTransformation()
    train_arr,test_arr,obj_file_path=obj3.inititate_data_transformation()
   
    obj4=ModelTrainer()
    print(obj4.initiate_model_trainer(train_arr,test_arr))

    #obj5=ModelEvaluation()
    #print(obj5.initiate_model_evaluation(train_arr,test_arr))




