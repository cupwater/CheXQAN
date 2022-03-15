<!--
 * @Author: Baoyun Peng
 * @Date: 2022-03-15 13:33:26
 * @LastEditTime: 2022-03-15 13:42:26
 * @Description: 
 * 
-->
`ai_model_center`: 这张表，主要是放模型标识的，就比如这次的总任务是“胸部DR正位”，在这个表里就是一行数据、但是这张表是手动填进去的，涉及到对模型的编码，需要问产品经理要，他应该有一套编码的方式。我正在问他要，他给我以后我就直接把信息插入表内.
![ai_model_center](/db/imgs/1.png "Text to show on mouseover").


`ai_model_template_module_info`：这张表是用来直接给前端做展示的，我们填，后端来读。header_title：质控模板里的大标题，比如“影像伪影显示评价”“整体印象评价”之类的；title：小标题，即具体的AI任务，比如“影像是否不存在异物伪影”；“content”：模型对每个AI任务的预测结果，比如“是/否”。“score”：根据预测结果给的分数，这个可以根据我们的算分逻辑来填，比如“0/1”.
![ai_model_center](/db/imgs/3.png "Text to show on mouseover").



`ai_model_data_center`：这张表后端称为“AI质控模型数据中心”，以下字段是后端填进来的，供我们读取，读完以后再写入另一张表
![ai_model_center](/db/imgs/2.png "Text to show on mouseover").

`ai_model_finish_template_info`（但是这些信息对于推理来说没什么用，主要是方便他们取数据处理）
![ai_model_center](/db/imgs/4.png "Text to show on mouseover").


task_id：这个字段存的是`质控任务`的标识，现阶段一个质控任务就是一个医院每天定时传上来几百张图片准备过AI模型进行推理，不是我们说的`AI任务`；
model_unique_code：这个字段是产品经理给出以后，我们手动存在`ai_model_center`这张表里的，后台会读走，然后填入这里；
system_source：系统来源；如：1-区域质控 2-云质控 3-生态方', 这个不需要我们管，目前只有区域质控；
hospital_code：医院code；
data_time：数据时间；数据上传的时间；
study_primary_id：影像唯一id，目前我们训练的时候使用 seriers_instance_uid作为影像唯一id的，但是在一期质控规划里只考虑了胸部DR正位，所以`study_primary_id` 和`seriers_instance_uid`是一一对应关系（在多部位，多体位和CT中，是一对多的关系），可以直接用来作为影像的唯一标识；
file_path：影像在数据库里的存储地址，与我们无关；
template_id：模板的唯一标识；一个模板包含了多个ai任务，比如在本期工作里，`胸部正位DR`质控模板包含了20个AI任务。以下是在该表中我们需要填入的内容：
ai_score：这个ai分数是指多个ai任务推理后进行计算的总分，通过后台传来的`study_primary_id` 来对应影像与ai分数；
ai_score_level：ai分数等级，如A,B,C,D；同样，也是按总分算出来的等级，目前没有换算标准，我需要问产品经理有没有规划；
state：状态：1-待推理 2-推理成功 3-推理失败 4-发生异常；

