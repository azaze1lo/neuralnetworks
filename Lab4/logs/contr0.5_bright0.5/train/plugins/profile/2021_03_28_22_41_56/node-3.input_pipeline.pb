  *`??"[x?@/?$?o?@)      ?=2?
`Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map::MapP@???@!?>???I@)????i@1蔜H??H@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2??!???)@!?)\vfV@)??
??@11΋??<@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map?r/0+\!@!?Pב?$N@)d${??!??1I??6$@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map::Map::ParallelMapV2::FlatMap[0]::TFRecord??L?D???!t??V?#@)??L?D???1t??V?#@:Advanced file read2?
oIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map::Map::ParallelMapV2?%?"???!GlꏦF??)?%?"???1GlꏦF??:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl.8??_t!@!?ZHp ON@)?y?S?4??1?߄8???:Preprocessing2?
xIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map::Map::ParallelMapV2::FlatMap??:????!???N?#@)T?YO???1?o?????:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch!;oc?#??!??l?Z??)!;oc?#??1??l?Z??:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCache?S??!@!#kruhN@)??66;R??1*??"u??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake???_Zԯ?!f?8?????)&W??Ma??1??ڦ????:Preprocessing2F
Iterator::Modelaũ??,??!#??????)S?r/0+??1KCf????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?n??\???!?gh ????)ȷw??{?1{?{C(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.