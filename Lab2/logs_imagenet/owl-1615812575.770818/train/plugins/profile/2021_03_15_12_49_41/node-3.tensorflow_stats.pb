"? 
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
bHost
DecodeJpeg"
DecodeJpeg(1!?rh???@9E"GM?s?@A!?rh???@IE"GM?s?@an#A????in#A?????Unknown
BHostIDLE"IDLE1?????@A?????@a??G'@???il#	?????Unknown
dHostCast"convert_image/Cast(1?/???@9??92??@A?/???@I??92??@a?[?????iP?L6 ????Unknown
^HostMul"convert_image(1V-??Ʒ@9?!???J?@AV-??Ʒ@I?!???J?@a?1aGy???ik??ʷ???Unknown
qHostResizeBilinear"resize/ResizeBilinear(1?G?Z(?@9LZŤU?~@A?G?Z(?@ILZŤU?~@a?]?9~R??iZ؎?K????Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(1V-?]?@9V-?]k@AV-?]?@IV-?]k@a?s ;???i??g??????Unknown
?HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(1???M"G?@9???M"G`@A???M"G?@I???M"G`@a??L??i?v???????Unknown
n	HostSub"per_image_standardization/sub(1B`?Тd?@9B`?ТdW@AB`?Тd?@IB`?ТdW@a?????P??i?nvk+D???Unknown
?
HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(1?z?G??@9?z?G?Q@A?z?G??@I?z?G?Q@a??jn¡z?iwDS?ny???Unknown
pHostMean"per_image_standardization/Mean(1-???'?@9-???'P@A-???'?@I-???'P@a??M??w?i{z?n?????Unknown
nHostRealDiv"per_image_standardization(1-??律?@9-??律M@A-??律?@I-??律M@a5?%SS?u?i_Ɣ?????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(1!?rh??s@9!?rh??3@A!?rh??s@I!?rh??3@aJ??2?\?i???.K????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(1?????f@9?????&@A??C?\X@I??C?\@a???\B?i???E?????Unknown
[HostOneHot"one_hot(1/?$?]U@9/?$?]@A/?$?]U@I/?$?]@ag?-???iØ?k?????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(1?Zd;U@9?Zd;@A?Zd;U@I?Zd;@a???n??ieI9A?????Unknown
eHost
LogicalAnd"
LogicalAnd(15^?IS@95^?IS@A5^?IS@I5^?IS@aa?h?#<?i?L?3????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1?v??oG@9?v??oG@A?v??oG@I?v??oG@a_@r??X1?i?Y?^????Unknown?
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(1?p=
W??@9?p=
W?`@Aw??/?F@Iw??/?@aM??`?0?i??[?t????Unknown
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1F???ԘC@9F???ԘC@AF???ԘC@IF???ԘC@a?i??-?i???*E????Unknown
iHostWriteSummary"WriteSummary(1X9??:@9X9??:@AX9??:@IX9??:@aQby??O#?iՔ6&z????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1?5^?I?6@9?5^?I?6@A?5^?I?6@I?5^?I?6@a????? ?i?L???????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(1?(\??Mi@9?(\??M)@Au?V4@Iu?V??a	g_???iHu*w????Unknown
vHostMaximum"!per_image_standardization/Maximum(1D?l??i2@9D?l??i??AD?l??i2@ID?l??i??aG????B?i????Q????Unknown
{HostSqrt")per_image_standardization/reduce_std/Sqrt(1????̌+@9????̌??A????̌+@I????̌??a??&??d?i?rd?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1?Zd;OJ@9?Zd;OJ@A??????*@I??????*@aO?#c???i%??c?????Unknown
lHostIteratorGetNext"IteratorGetNext(1V-?@9V-?@AV-?@IV-?@a?[x????i??F?????Unknown
dHostDataset"Iterator::Model(1Zd;?wP@9Zd;?wP@Ah??|?5@Ih??|?5@a???#?#?iv%??>????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1+??M@9+??M@A+???@I+???@a;?wv?i?{??????Unknown
eHost_Send"IteratorGetNext/_1(1?Zd;@9?Zd;@A?Zd;@I?Zd;@aA"?.???>i?`?D?????Unknown
eHost_Send"IteratorGetNext/_3(11?Zd@91?Zd@A1?Zd@I1?Zd@a??tV??>im?C??????Unknown
a HostIdentity"Identity(1+??????9+??????A+??????I+??????a???$?8?>iF?3?????Unknown?
?!Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1?5^?I??9?5^?I??A?5^?I??I?5^?I??a??????>i      ???Unknown*?
bHost
DecodeJpeg"
DecodeJpeg(1!?rh???@9E"GM?s?@A!?rh???@IE"GM?s?@a?1/I????i?1/I?????Unknown
dHostCast"convert_image/Cast(1?/???@9??92??@A?/???@I??92??@a??0?7???i?h;*G????Unknown
^HostMul"convert_image(1V-??Ʒ@9?!???J?@AV-??Ʒ@I?!???J?@a?Xo???iR?,??U???Unknown
qHostResizeBilinear"resize/ResizeBilinear(1?G?Z(?@9LZŤU?~@A?G?Z(?@ILZŤU?~@a??~?9??i0?$?x????Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(1V-?]?@9V-?]k@AV-?]?@IV-?]k@aLm%픘?i???% N???Unknown
?HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(1???M"G?@9???M"G`@A???M"G?@I???M"G`@a??????if?mp????Unknown
nHostSub"per_image_standardization/sub(1B`?Тd?@9B`?ТdW@AB`?Тd?@IB`?ТdW@a?]e?@??i?܆?t???Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(1?z?G??@9?z?G?Q@A?z?G??@I?z?G?Q@a F?X??i?????Z???Unknown
p	HostMean"per_image_standardization/Mean(1-???'?@9-???'P@A-???'?@I-???'P@a???sZ.}?id?nf1????Unknown
n
HostRealDiv"per_image_standardization(1-??律?@9-??律M@A-??律?@I-??律M@a?????z?i?????????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(1!?rh??s@9!?rh??3@A!?rh??s@I!?rh??3@aCmn[?a?i//??????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(1?????f@9?????&@A??C?\X@I??C?\@a?_?0"F?i?RL????Unknown
[HostOneHot"one_hot(1/?$?]U@9/?$?]@A/?$?]U@I/?$?]@a?2=?LiC?ii?v?&????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(1?Zd;U@9?Zd;@A?Zd;U@I?Zd;@aM7?NJC?i.$f?????Unknown
eHost
LogicalAnd"
LogicalAnd(15^?IS@95^?IS@A5^?IS@I5^?IS@a????EA?iVP?J????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1?v??oG@9?v??oG@A?v??oG@I?v??oG@a$3V wJ5?i[???????Unknown?
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(1?p=
W??@9?p=
W?`@Aw??/?F@Iw??/?@aF??4}4?i?????????Unknown
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1F???ԘC@9F???ԘC@AF???ԘC@IF???ԘC@aC6-??1?ig?g^?????Unknown
iHostWriteSummary"WriteSummary(1X9??:@9X9??:@AX9??:@IX9??:@a???{??'?iU?/?8????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1?5^?I?6@9?5^?I?6@A?5^?I?6@I?5^?I?6@a?????$?i???ׅ????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(1?(\??Mi@9?(\??M)@Au?V4@Iu?V??a$z??8"?i???d?????Unknown
vHostMaximum"!per_image_standardization/Maximum(1D?l??i2@9D?l??i??AD?l??i2@ID?l??i??a?˼ߺ ?ioW??????Unknown
{HostSqrt")per_image_standardization/reduce_std/Sqrt(1????̌+@9????̌??A????̌+@I????̌??a?Ǒ??i??P}????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1?Zd;OJ@9?Zd;OJ@A??????*@I??????*@a??.U?d?it??x@????Unknown
lHostIteratorGetNext"IteratorGetNext(1V-?@9V-?@AV-?@IV-?@a?Gф??i????????Unknown
dHostDataset"Iterator::Model(1Zd;?wP@9Zd;?wP@Ah??|?5@Ih??|?5@a'???@??iN.??????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1+??M@9+??M@A+???@I+???@a??(
?r?i?V_?p????Unknown
eHost_Send"IteratorGetNext/_1(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a ????O?>iP1R?????Unknown
eHost_Send"IteratorGetNext/_3(11?Zd@91?Zd@A1?Zd@I1?Zd@a??^OX??>i?????????Unknown
aHostIdentity"Identity(1+??????9+??????A+??????I+??????a??F?ח?>i!ƹ&?????Unknown?
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1?5^?I??9?5^?I??A?5^?I??I?5^?I??a?u?9F??>i      ???Unknown2GPU