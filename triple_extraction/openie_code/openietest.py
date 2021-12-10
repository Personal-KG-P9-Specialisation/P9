from openie import StanfordOpenIE

properties = {
    'openie.affinity_probability_cap': 2 / 3,
}
def annotate(client,text):
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-',triple)
with StanfordOpenIE(properties=properties) as client:
    text = 'I have neighbors. My neighbors dog won\'t stop barking at me. Ugh!'
    annotate(client, text)
    annotate(client, 'It\'s a little dog. I like big dogs. Why is it the little ones always bark the most?')
    annotate(client, 'I\'d like to introduce that dog to my pet snakes. I think they\'d eat him though!')
    annotate(client, 'My snakes are both pythons. I feed them mice. Do you have any pets?')
