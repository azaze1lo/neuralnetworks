"? 
DDeviceIDLE"IDLE1???????>A???????>Q      ??Y      ???Unknown
BHostIDLE"IDLE1?G?z=kAA?G?z=kAa?S]?3??i?S]?3???Unknown
bHost
DecodeJpeg"
DecodeJpeg( 1H?z??A9H?z?ƻ@AH?z??AIH?z?ƻ@aZ{????in?+?????Unknown
dHostCast"convert_image/Cast(1?Mb???@9v?Zjnӣ@A?Mb???@Iv?Zjnӣ@a٭?M.???i?cL??S???Unknown
^HostMul"convert_image(1ˡE??8?@9???9??@AˡE??8?@I???9??@a3??????i ??>???Unknown
?HostDataset"?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[4]::TFRecord(!1+??@9???zT?@A+??@I???zT?@a???L ??ix&<?????Unknown
qHostResizeBilinear"resize/ResizeBilinear(1    ???@9#,?4?@A    ???@I#,?4?@a??k?e??i]???D????Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(!1?????@9?r???bW@A?????@I?r???bW@aGVd
2m?i?[??v????Unknown
n	HostSub"per_image_standardization/sub(!1??(\??@9׵f]k&O@A??(\??@I׵f]k&O@a?hu?qc?i????????Unknown
?
HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(!1??x?&P?@9?im=]N@A??x?&P?@I?im=]N@a?f?b?i8ә??????Unknown
pHostMean"per_image_standardization/Mean(!1?5^?I@9?i?80?K@A?5^?I@I?i?80?K@a]7?Nha?io?\?D????Unknown
nHostRealDiv"per_image_standardization(!1ˡE?sV?@9??,c*?E@AˡE?sV?@I??,c*?E@av^#?
[?i??nB?????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(!133333?~@9?oX???-@A33333?~@I?oX???-@a?l??e?B?is?m????Unknown
[HostOneHot"one_hot(!1????r@9??!@A????r@I??!@a?V??5?ih3ԃ(????Unknown
?HostDataset"EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch(1?v??cb@9?v??cb@A?v??cb@I?v??cb@a^ઢOB&?i^Ψ?????Unknown
eHost_Send"IteratorGetNext/_1(1T㥛??X@9T㥛??X@AT㥛??X@IT㥛??X@a?2?i?dJ}????Unknown
?HostDataset"cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl( 1??C?|b@9??C?|@AV-???W@IV-???@a+"r????i?"
d????Unknown
?HostDataset"{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(!1Zd;߇#?@9????Eh?@A?MbXiT@I14? ?@a???/???i??{,*????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????O@9??????O@A??????O@I??????O@a??P?GG?iA[?f?????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1??????M@9??????M@A??????M@I??????M@aH2????i;`&U????Unknown?
eHost_Send"IteratorGetNext/_3(1Zd;?_J@9Zd;?_J@AZd;?_J@IZd;?_J@a??I????icʼ?????Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2( 1m????2J@9m????2??Am????2J@Im????2??aA?$dE??i??ߙS????Unknown
iHostWriteSummary"WriteSummary(1D?l??YC@9D?l??YC@AD?l??YC@ID?l??YC@a???-m?iu2?N?????Unknown?
eHost_Send"IteratorGetNext/_6(1y?&1A@9y?&1A@Ay?&1A@Iy?&1A@a???????iU)??????Unknown
?HostDataset"_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache( 11?Zdf@91?Zd@AX9??v>?@IX9??v>??a?B?`??i_??O????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_9(1Zd;?O6@9Zd;?O6@AZd;?O6@IZd;?O6@a[?}??>i?????????Unknown
?HostDataset";Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle(1??K7?id@9??K7?id@A?~j?t30@I?~j?t30@a??lmМ?>i????????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1?S㥛?f@9?S㥛?f@AD?l??)'@ID?l??)'@a???
?>i?7?(?????Unknown
dHostDataset"Iterator::Model(1B`??"?g@9B`??"?g@A7?A`??@I7?A`??@a?^??>i??4?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1D?l??Qe@9D?l??Qe@Au?V@Iu?V@ad??V??>ik??????Unknown
lHostIteratorGetNext"IteratorGetNext(1?/?$?@9?/?$?@A?/?$?@I?/?$?@a?OQ?Dz?>i+?'?????Unknown
a HostIdentity"Identity(1???Q?@9???Q?@A???Q?@I???Q?@a?ZSa??>i      ???Unknown?*?
bHost
DecodeJpeg"
DecodeJpeg( 1H?z??A9H?z?ƻ@AH?z??AIH?z?ƻ@a?)?A?-??i?)?A?-???Unknown
dHostCast"convert_image/Cast(1?Mb???@9v?Zjnӣ@A?Mb???@Iv?Zjnӣ@a6?S?Q???iPb?????Unknown
^HostMul"convert_image(1ˡE??8?@9???9??@AˡE??8?@I???9??@a???????i???Z????Unknown
?HostDataset"?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[4]::TFRecord(!1+??@9???zT?@A+??@I???zT?@ah????}??i?HC3????Unknown
qHostResizeBilinear"resize/ResizeBilinear(1    ???@9#,?4?@A    ???@I#,?4?@a??????i\xS
???Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(!1?????@9?r???bW@A?????@I?r???bW@a;?Z????i??H?3I???Unknown
nHostSub"per_image_standardization/sub(!1??(\??@9׵f]k&O@A??(\??@I׵f]k&O@a????u?i?|??>s???Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(!1??x?&P?@9?im=]N@A??x?&P?@I?im=]N@a???g~t?i
`?;????Unknown
p	HostMean"per_image_standardization/Mean(!1?5^?I@9?i?80?K@A?5^?I@I?i?80?K@a.p?)?r?i???l?????Unknown
n
HostRealDiv"per_image_standardization(!1ˡE?sV?@9??,c*?E@AˡE?sV?@I??,c*?E@aqGl?<m?i1	?+????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(!133333?~@9?oX???-@A33333?~@I?oX???-@a?????T?i{e??"????Unknown
[HostOneHot"one_hot(!1????r@9??!@A????r@I??!@a?????G?i|????Unknown
?HostDataset"EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch(1?v??cb@9?v??cb@A?v??cb@I?v??cb@a"áZ8?i?a?#????Unknown
eHost_Send"IteratorGetNext/_1(1T㥛??X@9T㥛??X@AT㥛??X@IT㥛??X@aD?ݵ?B0?in {????Unknown
?HostDataset"cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl( 1??C?|b@9??C?|@AV-???W@IV-???@a? zY?/?i|??p	????Unknown
?HostDataset"{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(!1Zd;߇#?@9????Eh?@A?MbXiT@I14? ?@a?-f>T?*?iߥ???????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????O@9??????O@A??????O@I??????O@a????$?i-TJf????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1??????M@9??????M@A??????M@I??????M@a?8??#?i?w?!;????Unknown?
eHost_Send"IteratorGetNext/_3(1Zd;?_J@9Zd;?_J@AZd;?_J@IZd;?_J@aG,?ƮB!?i@??LO????Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2( 1m????2J@9m????2??Am????2J@Im????2??a,?0?G%!?iKH3?a????Unknown
iHostWriteSummary"WriteSummary(1D?l??YC@9D?l??YC@AD?l??YC@ID?l??YC@a<?p'T?i?ckB,????Unknown?
eHost_Send"IteratorGetNext/_6(1y?&1A@9y?&1A@Ay?&1A@Iy?&1A@a??j?O?i??>??????Unknown
?HostDataset"_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache( 11?Zdf@91?Zd@AX9??v>?@IX9??v>??a]X??r?i??>V?????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_9(1Zd;?O6@9Zd;?O6@AZd;?O6@IZd;?O6@a?+????iǋN??????Unknown
?HostDataset";Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle(1??K7?id@9??K7?id@A?~j?t30@I?~j?t30@aք?Ɂ4?i??U?J????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1?S㥛?f@9?S㥛?f@AD?l??)'@ID?l??)'@a	#??Q?>i?????????Unknown
dHostDataset"Iterator::Model(1B`??"?g@9B`??"?g@A7?A`??@I7?A`??@a??)B??>i?CF?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1D?l??Qe@9D?l??Qe@Au?V@Iu?V@a??S??>i???M?????Unknown
lHostIteratorGetNext"IteratorGetNext(1?/?$?@9?/?$?@A?/?$?@I?/?$?@a?_v?=??>i?/?????Unknown
aHostIdentity"Identity(1???Q?@9???Q?@A???Q?@I???Q?@aB??e???>i     ???Unknown?2GPU