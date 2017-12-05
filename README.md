# NLPProject

Trabalho final da disciplina **Representações distribuídas de texto e modelagem de tópicos**.

Professor: Renato Rocha Souza

# Atividade
Escolher um dos assuntos abordados na disciplina e aplicar em dados.

# Proposta
Utilizar a técnica de modelagem de tópicos nas publicações dos três últimos anos da revista Perspectivas em [Ciência da Informação (PCI)](http://portaldeperiodicos.eci.ufmg.br/index.php/pci) e identificar os principais assuntos abordados.

# Implementação
A linguagem escolhida no desenvolvimento foi python e as bibliotecas utilizadas foram:
* [NLTK](http://www.nltk.org)
* [Gensim](https://radimrehurek.com/gensim)

O conteúdo dos artigos foram armazenados, em texto plano, no diretório **docs**.

Para encontrar os 10 principais tópicos, foi utilizado o Latent Dirichlet Allocation (LDA) implementado na biblioteca [Gensim](https://radimrehurek.com/gensim).

A saída, contendo o número de tópicos e assuntos, é uma lista e corresponde à seguinte estrutura:

```sh
[(0, [(termo1, probabilidade_termo1), (termo2, probabilidade_termo2), ..., (termoN, probabilidade_termoN)]), (1, [(termo1, probabilidade_termo1), (termo2, probabilidade_termo2), ..., (termoN, probabilidade_termoN)]), (t, [(termo1, probabilidade_termo1), (termo2, probabilidade_termo2), ..., (termoN, probabilidade_termoN)])]
```

Onde: 
* t = número de tópicos desejados
* N = número de assuntos de cada tópico

Assim, um exemplo de saída para 10 tópicos e 10 assuntos em cada tópico é:
```sh
[(0, [('informação', 0.00018429906026683684), ('conhecimento', 0.00018427798776991409), ('científica', 0.00018426530007554023), ('forma', 0.00018426250302630258), ('ciência', 0.00018425785660894825), ('b', 0.00018425090463782706), ('produção', 0.00018424963747390642), ('dados', 0.00018424842111267758), ('histórias', 0.00018424805406731963), ('citações', 0.00018424769462130794)]),
(1, [('periódicos', 0.00018432074673330629), ('ciência', 0.00018429878719721773), ('científica', 0.00018429066647836575), ('informação', 0.00018428858948845957), ('conhecimento', 0.00018428772091068674), ('dados', 0.00018428007379528186), ('forma', 0.0001842771136030013), ('pesquisa', 0.00018427594374678514), ('revistas', 0.00018426730316101404), ('quantidade', 0.00018426703682866)]),
(2, [('científica', 0.010077071501879429), ('ciência', 0.0064866640154857772), ('fi', 0.0060079450877829109), ('citações', 0.0057685850512283424), ('artigos', 0.0055292243917214948), ('dados', 0.0052898637599530698), ('forma', 0.0050505034925067919), ('cientistas', 0.00457178329508317), ('reconhecimento', 0.0045717830235777728), ('científico', 0.0045717830078323366)]),
(3, [('periódicos', 0.022025017252380843), ('b', 0.01713599871613582), ('revistas', 0.0095580246885691739), ('área', 0.0085802232267920206), ('administração', 0.0076024191183253478), ('conhecimento', 0.0068690699366047913), ('publicação', 0.0058912633896305612), ('região', 0.0056468119801633566), ('ciências', 0.0051579106846817591), ('estrato', 0.0049134602962849034)]),
(4, [('histórias', 0.016206920666405982), ('voz', 0.0074935230467189293), ('texto', 0.0059997984083965837), ('livro', 0.0057508438345759451), ('leitura', 0.005750843634760412), ('leitor', 0.0052529352063187722), ('personagens', 0.0050039801868827852), ('ler', 0.0047550265435265865), ('oral', 0.0045060716029989203), ('contar', 0.0042571177193743478)]),
(5, [('informação', 0.00018429246013260331), ('conhecimento', 0.00018428925067717175), ('periódicos', 0.00018428787303186983), ('b', 0.00018426862850699392), ('área', 0.00018425762166576317), ('revistas', 0.00018425644255309608), ('histórias', 0.00018425465543720179), ('ciência', 0.00018425164610674297), ('dados', 0.0001842467748534232), ('ainda', 0.00018424602720701623)]),
(6, [('periódicos', 0.00018432790706987543), ('b', 0.00018430479214465474), ('conhecimento', 0.00018430391968683561), ('informação', 0.00018429706093542011), ('revistas', 0.00018427279035614273), ('dados', 0.0001842700058297257), ('publicação', 0.00018426934395105802), ('área', 0.00018426864346711254), ('administração', 0.00018426174423410578), ('forma', 0.00018426103661657337)]),
(7, [('dispositivos', 0.011241951677325897), ('informação', 0.010268641538337871), ('móveis', 0.0092952938557534157), ('forma', 0.0063753092495243387), ('aprendizagem', 0.0058886423027211237), ('dados', 0.0049153162655340499), ('realidade', 0.0049153132979664364), ('conhecimento', 0.004428663786443035), ('uso', 0.0044286502727590759), ('contexto', 0.0044286502577764045)]),
(8, [('informação', 0.00018425545759360082), ('conhecimento', 0.00018425330188095022), ('científica', 0.00018424590534637324), ('ciência', 0.0001842429415375061), ('periódicos', 0.00018424210193494151), ('forma', 0.00018424119237914281), ('comunicação', 0.00018423986796559279), ('dados', 0.00018423941742173213), ('histórias', 0.00018423931258040654), ('científico', 0.00018423801061393167)]),
(9, [('informação', 0.035810903703249027), ('conhecimento', 0.0198748987740822), ('mídias', 0.0063443317257188278), ('ciência', 0.0060436556785546181), ('paradigma', 0.0060436524547269419), ('comunicação', 0.0051416161273873393), ('buckland', 0.0051416150208916664), ('zins', 0.0042395775568725962), ('processo', 0.0039388987498421915), ('tecnologias', 0.00393889846990674)])]
```