
Construir GitHub Documentación Lanzamiento de GitHub Pacto de contribuyentes

Procesamiento de lenguaje natural de última generación para PyTorch y TensorFlow 2.0

🤗Transformers proporciona miles de modelos previamente entrenados para realizar tareas en textos como clasificación, extracción de información, respuesta a preguntas, resumen, traducción, generación de texto, etc. en más de 100 idiomas. Su objetivo es hacer que la PNL de vanguardia sea más fácil de usar para todos.

🤗Transformers proporciona API para descargar y usar rápidamente esos modelos previamente entrenados en un texto dado, ajustarlos en sus propios conjuntos de datos y luego compartirlos con la comunidad en nuestro centro de modelos . Al mismo tiempo, cada módulo de Python que define una arquitectura se puede usar de forma independiente y se puede modificar para permitir experimentos de investigación rápidos.

🤗Transformers está respaldado por las dos bibliotecas de aprendizaje profundo más populares, PyTorch y TensorFlow , con una integración perfecta entre ellas, lo que le permite entrenar sus modelos con uno y luego cargarlo para inferencia con el otro.

Demos en línea
Puede probar la mayoría de nuestros modelos directamente en sus páginas desde el centro de modelos . También ofrecemos una API de inferencia para usar esos modelos.

Aquí están algunos ejemplos:

Completar palabras enmascaradas con BERT
Reconocimiento de entidad de nombre con Electra
Generación de texto con GPT-2
Inferencia de lenguaje natural con RoBERTa
Resumen con BART
Respuesta a preguntas con DistilBERT
Traducción con T5
Write With Transformer , creado por el equipo Hugging Face, es la demostración oficial de las capacidades de generación de texto de este repositorio.

Tour rapido
Para usar inmediatamente un modelo en un texto dado, proporcionamos la pipelineAPI. Las canalizaciones agrupan un modelo previamente entrenado con el procesamiento previo que se utilizó durante ese entrenamiento del modelo. A continuación, se explica cómo utilizar rápidamente una canalización para clasificar textos positivos y negativos

>> >  de  transformadores de  importación  de tuberías

# Asignar una tubería para el sentimiento-análisis 
>> >  clasificador  =  tubería ( 'sentimiento-análisis' )
 >> >  clasificador ( 'Estamos muy contentos de incluir la tubería en el depósito de transformadores.' )
[{ 'label' : 'POSITIVO' , 'puntuación' : 0,9978193640708923 }]
La segunda línea de código descarga y almacena en caché el modelo preentrenado utilizado por la canalización, la tercera línea lo evalúa en el texto dado. Aquí la respuesta es "positiva" con una confianza del 99,8%.

Este es otro ejemplo de canalización que puede extraer respuestas a preguntas de algún contexto:

>> >  de  transformadores de  importación  de tuberías

# Asignar un gasoducto para la pregunta-respuesta 
>> >  question_answerer  =  tubería ( 'de pregunta-respuesta' )
 >> >  question_answerer ({
...      'pregunta' : '¿Cuál es el nombre del repositorio?' ,
...      'context' : 'Pipeline ha sido incluido en el repositorio huggingface / transformers'
...})
{ 'score' : 0.5135612454720828 , 'start' : 35 , 'end' : 59 , 'answer' : 'huggingface / transformers' }
Además de la respuesta, el modelo preentrenado utilizado aquí devolvió su puntuación de confianza, junto con la posición inicial y la posición final en la oración tokenizada. Puede obtener más información sobre las tareas compatibles con la pipelineAPI en este tutorial .

Para descargar y usar cualquiera de los modelos previamente entrenados en su tarea dada, solo necesita usar esas tres líneas de códigos (versión de PyTorch):

>> >  de  transformadores  importar  AutoTokenizer , Automodel

>> >  tokenizer  =  AutoTokenizer . from_pretrained ( "Bert-base-entubar" )
 >> >  modelo  =  Automodel . from_pretrained ( "bert-base-uncased" )

>> >  entradas  =  tokenizer ( "Hola mundo!" , Return_tensors = "PT" )
 >> >  salidas  =  modelo ( ** entradas )
o para TensorFlow:

>> >  de  transformadores  importar  AutoTokenizer , TFAutoModel

>> >  tokenizer  =  AutoTokenizer . from_pretrained ( "Bert-base-entubar" )
 >> >  modelo  =  TFAutoModel . from_pretrained ( "bert-base-uncased" )

>> >  entradas  =  tokenizer ( "Hola mundo!" , Return_tensors = "TF" )
 >> >  salidas  =  modelo ( ** entradas )
El tokenizador es responsable de todo el preprocesamiento que espera el modelo preentrenado, y se puede llamar directamente en uno (o lista) de textos (como podemos ver en la cuarta línea de ambos ejemplos de código). Generará un diccionario que puede pasar directamente a su modelo (que se hace en la quinta línea).

El modelo en sí es un Pytorchnn.Module normal o un TensorFlowtf.keras.Model (dependiendo de su backend) que puede usar normalmente. Por ejemplo, este tutorial explica cómo integrar un modelo de este tipo en el ciclo de entrenamiento clásico de PyTorch o TensorFlow, o cómo usar nuestra TrainerAPI para ajustar rápidamente en un nuevo conjunto de datos.

¿Por qué debería usar transformadores?
Modelos de última generación fáciles de usar:

Alto rendimiento en tareas NLU y NLG.
Barrera de entrada baja para educadores y profesionales.
Pocas abstracciones orientadas al usuario con solo tres clases para aprender.
Una API unificada para usar todos nuestros modelos previamente entrenados.
Menores costos de computación, menor huella de carbono:

Los investigadores pueden compartir modelos entrenados en lugar de siempre volver a capacitarse.
Los profesionales pueden reducir el tiempo de cálculo y los costos de producción.
Docenas de arquitecturas con más de 2000 modelos previamente entrenados, algunos en más de 100 idiomas.
Elija el marco adecuado para cada parte de la vida útil de un modelo:

Entrene modelos de última generación en 3 líneas de código.
Mueva un solo modelo entre marcos TF2.0 / PyTorch a voluntad.
Elija sin problemas el marco adecuado para la formación, la evaluación y la producción.
Personalice fácilmente un modelo o un ejemplo según sus necesidades:

Ejemplos de cada arquitectura para reproducir los resultados de los autores oficiales de dicha arquitectura.
Exponga los modelos internos de la forma más coherente posible.
Los archivos de modelo se pueden usar independientemente de la biblioteca para experimentos rápidos.
¿Por qué no debería usar transformadores?
Esta biblioteca no es una caja de herramientas modular de bloques de construcción para redes neuronales. El código de los archivos del modelo no se refactoriza con abstracciones adicionales a propósito, de modo que los investigadores puedan iterar rápidamente en cada uno de los modelos sin sumergirse en abstracciones / archivos adicionales.
La API de entrenamiento no está diseñada para funcionar en ningún modelo, pero está optimizada para funcionar con los modelos proporcionados por la biblioteca. Para bucles genéricos de aprendizaje automático, debe usar otra biblioteca.
Si bien nos esforzamos por presentar tantos casos de uso como sea posible, los scripts en nuestra carpeta de ejemplos son solo eso: ejemplos. Se espera que no funcionen de inmediato en su problema específico y que se le pedirá que cambie algunas líneas de código para adaptarlas a sus necesidades.
Instalación
Con pepita
Este repositorio se probó en Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ para ejemplos ) y TensorFlow 2.0.

Deberías instalar 🤗Transformadores en un entorno virtual . Si no está familiarizado con los entornos virtuales de Python, consulte la guía del usuario .

Primero, crea un entorno virtual con la versión de Python que vas a usar y actívalo.

Luego, deberá instalar al menos uno de TensorFlow 2.0, PyTorch o Flax. Por favor refiérase a la página de instalación TensorFlow , página de instalación PyTorch en relación con el comando de instalación específica para su plataforma y / o página de instalación de lino .

Cuando se haya instalado TensorFlow 2.0 y / o PyTorch, 🤗 Los transformadores se pueden instalar usando pip de la siguiente manera:

pip instalar transformadores
Si desea jugar con los ejemplos, debe instalar la biblioteca desde la fuente .

Con conda
Desde Transformers versión v4.0.0, ahora tenemos un canal Conda: huggingface.

🤗 Los transformadores se pueden instalar usando conda de la siguiente manera:

conda install -c huggingface transformers
Siga las páginas de instalación de TensorFlow, PyTorch o Flax para ver cómo instalarlas con conda.

Arquitecturas de modelos
Todos los puntos de control modelo proporcionados por🤗Los transformadores se integran a la perfección desde el centro de modelos huggingface.co , donde los usuarios y las organizaciones los cargan directamente .

Número actual de puntos de control: 

🤗Transformers actualmente proporciona las siguientes arquitecturas (consulte aquí un resumen de alto nivel de cada una):

ALBERT (de Google Research y el Instituto Tecnológico de Toyota en Chicago) publicado con el artículo ALBERT: A Lite BERT for Self-supervised Learning of Language Representations , por Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
BART (de Facebook) publicado con el documento BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension por Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov y Luke Zettlemoyer.
BARThez (de École polytechnique) publicado con el artículo BARThez: a Skilled Pretrained French Sequence-to-Sequence Model por Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
BERT (de Google) publicado con el documento BERT: Pre-formación de transformadores bidireccionales profundos para la comprensión del lenguaje por Jacob Devlin, Ming-Wei Chang, Kenton Lee y Kristina Toutanova.
BERT For Sequence Generation (de Google) publicado con el documento Aprovechando los puntos de control pre-entrenados para tareas de generación de secuencias por Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
Blenderbot (de Facebook) publicado con el papel Recetas para construir un chatbot de dominio abierto por Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
CamemBERT (de Inria / Facebook / Sorbonne) publicado con el artículo CamemBERT: a Tasty French Language Model de Louis Martin *, Benjamin Muller *, Pedro Javier Ortiz Suárez *, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah y Benoît Sagot.
CTRL (de Salesforce) publicado con el documento CTRL: A Conditional Transformer Language Model for Controllable Generation por Nitish Shirish Keskar *, Bryan McCann *, Lav R. Varshney, Caiming Xiong y Richard Socher.
DeBERTa (de Microsoft Research) publicado con el artículo DeBERTa: BERT mejorado con decodificación con atención desenredada de Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
DialoGPT (de Microsoft Research) publicado con el artículo DialoGPT: Entrenamiento previo generativo a gran escala para la generación de respuesta conversacional por Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
DistilBERT (de HuggingFace), publicado junto con el periódico DistilBERT, una versión destilada de BERT: más pequeño, más rápido, más barato y más ligero de Victor Sanh, Lysandre Debut y Thomas Wolf. Se ha aplicado el mismo método para comprimir GPT2 en DistilGPT2 , RoBERTa en DistilRoBERTa , BERT multilingüe en DistilmBERT y una versión alemana de DistilBERT.
DPR (de Facebook) publicado con el documento Recuperación de pasaje denso para respuesta a preguntas de dominio abierto por Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen y Wen-tau Yih.
ELECTRA (de Google Research / Stanford University) publicado con el artículo ELECTRA: Pre-entrenamiento de codificadores de texto como discriminadores en lugar de generadores por Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
FlauBERT (del CNRS) publicado con el artículo FlauBERT: Pre-formación del modelo lingüístico no supervisado para francés por Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
Funnel Transformer (de CMU / Google Brain) publicado con el artículo Funnel-Transformer: Filtrar la redundancia secuencial para un procesamiento eficiente del lenguaje por Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
GPT (de OpenAI) publicado con el artículo Improving Language Understanding by Generative Pre-Training por Alec Radford, Karthik Narasimhan, Tim Salimans e Ilya Sutskever.
GPT-2 (de OpenAI) publicado con el papel Los modelos de lenguaje son estudiantes multitarea no supervisados por Alec Radford *, Jeffrey Wu *, Rewon Child, David Luan, Dario Amodei ** e Ilya Sutskever **.
LayoutLM (de Microsoft Research Asia) publicado con el documento LayoutLM: Pre-training of Text and Layout for Document Image Understanding por Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
Longformer (de AllenAI) publicado con el artículo Longformer: The Long-Document Transformer por Iz Beltagy, Matthew E. Peters, Arman Cohan.
LXMERT (de UNC Chapel Hill) publicado con el documento LXMERT: Aprendizaje de representaciones de codificador de modalidades cruzadas de Transformers para la respuesta a preguntas de dominio abierto por Hao Tan y Mohit Bansal.
MarianMT Modelos de traducción automática entrenados condatos OPUS por Jörg Tiedemann. El Marco de Marian está siendo desarrollado por el traductor del equipo de Microsoft.
MBart (de Facebook) publicado con el artículo Multilingual Denoising Pre-training for Neural Machine Translation por Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
MPNet (de Microsoft Research) publicado con el documento MPNet: Pre-entrenamiento enmascarado y permutado para la comprensión del lenguaje por Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
MT5 (de Google AI) publicado con el documento mT5: un transformador de texto a texto masivamente multilingüe y previamente entrenado por Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel .
Pegasus (de Google) publicado con el artículo PEGASUS: Pre-training with Extracted Gap-oraciones para resumen abstracto > por Jingqing Zhang, Yao Zhao, Mohammad Saleh y Peter J. Liu.
ProphetNet (de Microsoft Research) publicado con el artículo ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training por Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang y Ming Zhou .
Reformer (de Google Research) publicado con el artículo Reformer: The Efficient Transformer de Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
RoBERTa (de Facebook), publicó junto con el artículo un enfoque de preentrenamiento BERT robustamente optimizado de Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. BERT ultilingüe en DistilmBERT y una versión alemana de DistilBERT.
SqueezeBert publicado con el documento SqueezeBERT: ¿Qué puede enseñar la visión por computadora a la PNL sobre redes neuronales eficientes? por Forrest N. Iandola, Albert E. Shaw, Ravi Krishna y Kurt W. Keutzer.
T5 (de Google AI) publicado con el artículo Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer por Colin Raffel y Noam Shazeer y Adam Roberts y Katherine Lee y Sharan Narang y Michael Matena y Yanqi Zhou y Wei Li y Peter J. Liu.
Transformer-XL (de Google / CMU) publicado con el documento Transformer-XL: Modelos de lenguaje atentos más allá de un contexto de longitud fija por Zihang Dai *, Zhilin Yang *, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
XLM (de Facebook) publicado junto con el documento Cross-lingual Language Model Pretraining por Guillaume Lample y Alexis Conneau.
XLM-ProphetNet (de Microsoft Research) publicado con el documento ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training por Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang y Ming Zhou.
XLM-RoBERTa (de Facebook AI), publicado junto con el documento Unsupervised Cross-lingual Representation Learning at Scale por Alexis Conneau *, Kartikay Khandelwal *, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer y Veselin Stoyanov.
XLNet (de Google / CMU) publicado con el artículo XLNet: Preentrenamiento autorregresivo generalizado para la comprensión del lenguaje por Zhilin Yang *, Zihang Dai *, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
¿Quieres aportar un nuevo modelo? Hemos agregado una guía detallada y plantillas para guiarlo en el proceso de agregar un nuevo modelo. Puedes encontrarlos en la templatescarpeta del repositorio. Asegúrese de verificar las pautas de contribución y comunicarse con los mantenedores o abrir un problema para recopilar comentarios antes de comenzar su PR.
Para verificar si cada modelo tiene una implementación en PyTorch / TensorFlow / Flax o tiene un tokenizador asociado respaldado por el 🤗Biblioteca de tokenizadores, consulte esta tabla

Estas implementaciones se han probado en varios conjuntos de datos (consulte los scripts de ejemplo) y deben coincidir con el rendimiento de las implementaciones originales. Puede encontrar más detalles sobre el rendimiento en la sección Ejemplos de la documentación .

Aprende más
Sección	Descripción
Documentación	Tutoriales y documentación completa de API
Resumen de la tarea	Tareas apoyadas por 🤗 Transformadores
Tutorial de preprocesamiento	Usar la Tokenizerclase para preparar datos para los modelos
Entrenamiento y puesta a punto	Usando los modelos proporcionados por 🤗Transformers en un ciclo de entrenamiento de PyTorch / TensorFlow y la TrainerAPI
Visita rápida: scripts de ajuste / uso	Scripts de ejemplo para ajustar modelos en una amplia gama de tareas
Compartir y cargar modelos	Sube y comparte tus modelos perfeccionados con la comunidad
Migración	Migrar a 🤗Transformadores de pytorch-transformersopytorch-pretrained-bert
Citación
Ahora tenemos un documento que puede citar para el🤗 Biblioteca de transformadores:

@inproceedings { wolf-etal-2020-transformers ,
     title = " Transformers: Procesamiento del lenguaje natural de última generación " ,
     autor = " Thomas Wolf y Lysandre Debut y Victor Sanh y Julien Chaumond y Clement Delangue y Anthony Moi y Pierric Cistac y Tim Rault y Rémi Louf y Morgan Funtowicz y Joe Davison y Sam Shleifer y Patrick von Platen y Clara Ma y Yacine Jernite y Julien Plu y Canwen Xu y Teven Le Scao y Sylvain Gugger y Mariama Drame y Quentin Lhoest y Alexander M. Rush " ,
     booktitle = "Actas de la Conferencia de 2020 sobre métodos empíricos en el procesamiento del lenguaje natural: demostraciones de sistemas " ,
     mes = oct,
     año = " 2020 " ,
     dirección = " Online " ,
     editor = " Association for Computational Linguistics " ,
     url = " https: // www .aclweb.org / anthology / 2020.emnlp-demos.6 " ,
     pages = " 38--45 " 
}
