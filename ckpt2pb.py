import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import optimize_for_inference_lib

def freeze_graph(input_checkpoint, output_graph):
    output_node_names = "input_dataset/input_dataset_x,fc2/add" #输出的节点
    input_node_names = 'input_dataset/input_dataset_x' #输入的节点
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        # 输出所有Layer的tensor
        # op = sess.graph.get_operations()
        # [print(m.values()) for m in op][1]

        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        # 压缩Graph，只保留输入输出
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(output_graph_def, [input_node_names], output_node_names.split(","), tf.float32.as_datatype_enum)
        output_graph_def = TransformGraph(output_graph_def, [input_node_names], output_node_names.split(","), ["sort_by_execution_order"])

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
 
if __name__ == '__main__':
    modelpath="./model/model.ckpt"
    freeze_graph(modelpath,"frozen.pb")
    print("finish!")
