import dash_bootstrap_components as dbc
from transformers import pipeline
from dash import Dash,Input,Output,dcc,html,State,dash_table
import requests
from textblob import TextBlob
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Get Quote

api_response = requests.get('https://favqs.com/api/qotd')
api_json = api_response.json()
quote_print = api_json['quote']['body']
author = api_json['quote']['author']
author_print = '-- '+author


card_quote = dbc.Card([quote_print,html.Br(),html.Br(),author_print,],className='card_quote')

# graphs

fig1=dcc.Graph(id = 'scatter_graph1', config = {'displayModeBar':False},figure = {})
fig2=dcc.Graph(id = 'bar_graph2', config = {'displayModeBar':False},figure = {})
fig3=dcc.Graph(id = 'pie_graph3', config = {'displayModeBar':False},figure = {})

none_graph = px.scatter(x = [0],y=[0],
                        title = 'No Data',
                        template='plotly_dark'
                        )

# tabs

tab_style = {'color':'#000000',                                               #style when tab is not selected
             'backgroundColor': '#909090',
                'fontWeight':'lighter',
              'fontSize':'1.0vw',
             'border-bottom': '1px solid black'}
selected_tab_style = {'color': '#000000',
                      'backgroundColor': '#E8E8E8',
                      'fontSize':'1.0vw',
                      'fontWeight':'bold',
                      'border-top':'1px solid black',
                      'border-left':'1px solid black',
                      'border-right':'1px solid black',
                      'border-bottom': '1px solid black'}
tab1 = dcc.Tab(fig1,
                label='Sentence Wise Sentiment/Polarity Value',
                style=tab_style,
                selected_style=selected_tab_style,
                value='fig1')
tab2 = dcc.Tab(fig2,
                label='Aggregate Values',
                style=tab_style,
                selected_style=selected_tab_style,
                value='fig2')
tab3 = dcc.Tab(fig3,
                label='Percentage distribution of Sentiment in Paragraph',
                style=tab_style,
                selected_style=selected_tab_style,
                value='fig3',
                )

tab_data = html.Div([

    dcc.Tabs(value = 'fig1', children = [tab1,tab2,tab3]),

], className='tabs_graph')

sentiment_table = dash_table.DataTable(

    id = 'datable_interactive'
)


textArea = dcc.Textarea(id = 'dcc_textarea',
                        placeholder='',
                        draggable=True,
                        spellCheck=True,
                        style = {'fontFamily':'Times New Roman','border-style':'solid','width':'100%','min-height':'88%','height':'auto','backgroundColor':'#FFFFFF','color':'black'})


textArea_question = dcc.Textarea(id = 'question',
                                 placeholder='Question',
                                 draggable=True,
                                 spellCheck=True,
                                 style = {'fontFamily':'Times New Roman','border-style':'solid','width':'100%','min-height':'20%','height':'auto'},
                                 className='question')

button = dbc.Container(
    [
        dbc.Button("GO!", id = 'button_id',color="success",outline=True, active=True, className="dbc_button",size='lm',n_clicks=0)
    ],className='button_container',fluid=True
)


app = Dash(meta_tags=[{'name':'viewport', 'content':'width=device-width'}],prevent_initial_callbacks=True)

navbar = dbc.NavbarSimple(children = [
    html.Img(src = 'https://cdn-icons-png.flaticon.com/512/945/945458.png?w=740&t=st=1678701855~exp=1678702455~hmac=6f067e5f6a47a0cd6810a7ffb2598c9a41046bb8bbdfc1aaa538ec3eeed80748',height = '50px',)
],
    brand='NLP Based QnA & Sentiment Analyzer',
    brand_style=  {'fontFamily':'Papyrus','fontSize':'40px',},
    style = {'margin-left':'5%','margin-right':'5%','margin-top':'5px',},
    dark=False,
)

para_quote = dbc.Row([
    dbc.Col(children=['Enter the Context Paragraph',html.Br(),
                      textArea],className='para_container',width={'size':10,'offset':0}),
    dbc.Col(
        dbc.Container(
        children=['A quote to Remember',html.Br(),card_quote], className='quote_container',fluid=True)),
], className='para_and_quote')

question_go = dbc.Row([
    dbc.Col(children = ['Ask your question below',html.Br(),textArea_question,html.Br()],className='ask_and_question',width={'size':10,'offset':0}),

    dbc.Col(children = [button],className='col_container')

],className='question_go_box')

answer_emoji = dbc.Row([
    dbc.Col(children = ['Your answer will appear below',
                        html.Br(),
                        html.Br(),
                        html.H5(id = 'answer_id',
                                children= [],
                                className='answer_value')],className='answer_class',width={'size':10,'offset':0}),
    dbc.Col(children=['Confidence Level',
                      html.Br(),
                      html.Br(),
                      html.H5(id='score_id',
                              children=[],
                              className='answer_value')],className='score_container'),
],className='answer_and_emoji')

nouns = dbc.Row([
    dbc.Col(children = ['Detected noun phrases will appear below',
                        html.Br(),
                        html.Br(),
                        html.H5(id = 'noun_id',
                                children = [],
                                className='noun_value')],className='answer_class',width={'size':10,'offset':0}),

    dbc.Col(children=['No. of Detected Noun Phrases',
                      html.Br(),
                      html.Br(),
                      html.H5(id='number_noun',
                              children=[],
                              className='answer_value')],className='score_container'),

],className='answer_and_emoji')

textblob_analysis = \
    html.Div([
        dbc.Row(children=['Sentence Wise Sentiment Analysis '],className='title_for_textblob'),

        dbc.Row([
            dbc.Col(children=['Sentences with Polarity and Sentiment',
                              html.Br(),
                              html.Br(),
                              sentiment_table,
                              ], className='table_col'),


            dbc.Col(children=['Graphical representation',
                              html.Br(),
                              html.Br(),
                              tab_data], className='table_col')

]),
        html.Br(),

    ], className='answer_and_emoji')

overall = html.Div([
    dbc.Row(children=['Overall Analysis'],className='title_for_textblob'),
    html.Br(),
    dbc.Row([
            dbc.Col(children = [html.Div('Sentiment',className='low_head'),html.Br(),html.Br(),html.Div(id = 'overall_sentiment',style = {'fontSize':'20px'})],className='overall_col'),
            dbc.Col(children=[html.Div('Polarity',className='low_head'),html.Br(),html.Br(),html.Div(id = 'overall_polarity',style = {'fontSize':'20px'})],className='overall_col'),
            dbc.Col(children=[html.Div('Subjectivity',className='low_head'),html.Br(),html.Br(),html.Div(id = 'overall_subjectivity',style = {'fontSize':'20px'})],className='overall_col'),
        ]),
    html.Br(),
],className='answer_and_emoji')

disclaimer = html.Div([

        'Disclaimer : This Dash App uses certain Python NLP packages, that may not produce 100% accurate results',

], className='disclaimer')


app.layout = html.Div([navbar,para_quote,question_go,answer_emoji,nouns,overall,textblob_analysis,html.Br(),html.Br(),disclaimer,html.Br(),html.Br()],className='main_container')
server = app.server

@app.callback(
    Output('answer_id','children'),
    Output('score_id', 'children'),
    Input('button_id', 'n_clicks'),
    State('dcc_textarea', 'value'),
    State('question', 'value'),
)
def update_answer(clicked,text_input,question_input):
    if clicked > 0 and text_input != '':
        type_nlp = pipeline('question-answering')
        ans = type_nlp(question=question_input, context=text_input)
        ans_final = ans['answer']
        score = str(round((ans['score'] * 100), 2)) + ' %'

        return ans_final,score

    else: return '',''

@app.callback(

    Output('noun_id', 'children'),
    Output('number_noun', 'children'),
    Output('overall_sentiment', 'children'),
    Output('overall_polarity', 'children'),
    Output('overall_subjectivity', 'children'),
    Input('button_id', 'n_clicks'),
    State('dcc_textarea', 'value'),

)

def update_others(clicked,text_input):

    if clicked > 0 and text_input != '' :

        blob = TextBlob(text_input)
        noun_set = (set(blob.noun_phrases))
        noun_list = list(noun_set)
        noun_text = noun_list[0]

        for i in range(1,len(noun_list)):
            noun_text = noun_text+' , '+noun_list[i]


        def getPolarity(text):
            return TextBlob(text).sentiment.polarity

        def getSubjectivity(text):
            return TextBlob(text).sentiment.subjectivity

        def getSentiment(text):
                p = TextBlob(text).sentiment.polarity
                if p>0 : return 'POSITIVE'
                elif p<0 : return 'NEGATIVE'
                else : return 'NEUTRAL'

        pol = np.round(getPolarity(text_input),2)
        sub = np.round(getSubjectivity(text_input),2)
        senten = getSentiment(text_input)


        return noun_text,str(len(noun_set))+' Unique Noun Phrases',senten,pol,sub

    else:
            return '','','','','','',''

@app.callback(
    Output('datable_interactive','data'),
    Output('datable_interactive','columns'),
    Output('datable_interactive','style_cell_conditional'),
    Output('datable_interactive', 'style_cell'),
    Output('datable_interactive', 'style_data'),
    Output('datable_interactive', 'page_current'),
    Output('datable_interactive', 'page_size'),
    Output('datable_interactive', 'sort_action'),
    Output('datable_interactive', 'sort_mode'),
    Output('datable_interactive', 'editable'),
    Output('datable_interactive', 'filter_action'),
    Output('datable_interactive', 'column_selectable'),
    Output('datable_interactive', 'row_selectable'),
    Output('datable_interactive', 'row_deletable'),
    Output('datable_interactive', 'selected_columns'),
    Output('datable_interactive', 'selected_rows'),

    Input('button_id', 'n_clicks'),
    State('dcc_textarea', 'value'),
)

def update_sentiment_table(clicked,text_input):

    if clicked>0 and text_input !='':

        blob = TextBlob(text_input)
        sent = (blob.sentences)
        list_of_sent = [str(i) for i in sent]
        df = pd.DataFrame(list_of_sent)
        df['Polarity'] = 0
        df['Sentiment'] = 'NA'
        df.rename(columns={0: 'Sentence'}, inplace=True)
        df.drop_duplicates(subset='Sentence', keep='first', inplace=True)
        df['Polarity'] = df['Sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df.loc[df.Polarity > 0, 'Sentiment'] = 'POSITIVE'
        df.loc[df.Polarity == 0, 'Sentiment'] = 'NEUTRAL'
        df.loc[df.Polarity < 0, 'Sentiment'] = 'NEGATIVE'
        df['id'] = ['S' + str(i) for i in range(1, len(df['Sentence']) + 1)]

        df['Polarity'] = np.round(df['Polarity'], 2)

        columns = [{'name': col, 'id': col} for col in df.columns]
        senti_data = df.to_dict(orient='records')
        #style_cell_conditional = [{'if':{'column_id':c},'textAlign':'left'}for c in ['Sentence','Sentiment','id','Polarity']]
        style_cell_conditional = [{'if':{'column_id':'Sentence'},'width':'56%'},
                                  {'if':{'column_id':'Sentiment'},'width':'14%'},
                                  {'if':{'column_id':'Polarity'},'width':'14%'},
                                  ]
        style_cell = {'minWidth':35,'maxWidth':35,'width':35,"whiteSpace": "pre-line",}
        style_data = {'whitespace':'normal','height':'auto','max-height':'200px'}
        page_current = 0
        page_size = 5
        sort_action = 'native'
        sort_mode = 'single'
        editable = True
        filter_action = 'native'
        column_selectable = 'multi'
        row_selectable = 'multi'
        row_deletable = True
        selected_columns = []
        selected_rows = []



        return senti_data, columns, style_cell_conditional,style_cell,style_data,page_current,page_size,sort_action, \
                sort_mode,editable,filter_action,column_selectable,row_selectable,row_deletable,selected_columns, \
                selected_rows



@app.callback(

    Output('scatter_graph1','figure'),
    Output('bar_graph2','figure'),
    Output('pie_graph3','figure'),


    Input('datable_interactive','derived_virtual_data'),
    Input('datable_interactive','derived_virtual_selected_rows')
)

def update_graphs(all_rows_data,selected_rows_indices):
    df_all = pd.DataFrame(all_rows_data)

    colors = ['#7FDBFF' if i in selected_rows_indices else '#000000' for i in range(len(df_all))]



    if 'id' in df_all:
        scatter_graph = {
        'data': [go.Scatter(
            x=df_all['id'],
            y=df_all['Polarity'],
            mode='text+markers',
            text=df_all['Polarity'],
            texttemplate='',
            marker=dict(color=colors, size=20, symbol='circle',line=dict(width=2, color='#000000')),
            textfont=dict(
                family='sans-serif',
                size=12,
                color='black'),
            textposition='bottom right',
            # line=dict(width=4, color='#318CE7'),
            hoverinfo='text',
            # name=(first_month),
            hovertext='<b>Sentiment :</b>' + df_all['Sentiment'].astype(str) + '<br>' +
                      '<b>Polarity :</b>' + df_all['Polarity'].astype(str) + '<br>'
        ), ],
        'layout': go.Layout(
            plot_bgcolor='#E8E8E8',
            paper_bgcolor='#E8E8E8',
            hovermode='closest',
            xaxis=dict(
                title='<b>Sentence Number</b>',
                visible=True,
                color='black',
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='black',
                linewidth=1,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='black'),
                tickmode='array',
                ticktext=df_all['id']
            ),
            yaxis=dict(
                title='<b>Polarity</b>',
                visible=True,
                color='black',
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='black',
                linewidth=1,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='black'),
            ),
            margin=dict(t=20, r=20),
            legend=dict(bgcolor='grey')
        ), }
# bar graph
        data_count = dict(df_all.pivot_table(index = ['Sentiment'], aggfunc ='size'))
        fig_bar = px.bar(
            x = list(data_count.keys()),
            y = list(data_count.values()),
            labels= {'x':'<b>Sentiment</b>','y':'<b>Total No. of Sentiment</b>'},

        )
        fig_bar.update_layout(plot_bgcolor='#E8E8E8',paper_bgcolor='#E8E8E8',)
        fig_bar.update_traces(marker_color='#000000',opacity = 0.7,marker_line_color='white',marker_line_width=2)
        fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig_bar.update_traces()

# pie graph
        data_count_df = pd.DataFrame([data_count]).T.reset_index()
        data_count_df.rename(columns={'index': 'Sentiment', 0: 'No. of Sentiment'}, inplace=True)
        fig_pie = px.pie(data_count_df,
                         values='No. of Sentiment',
                         names='Sentiment',
                         hole=0.5,
                         color = 'Sentiment',
                         color_discrete_map={'POSITIVE': '#000000',
                                             'NEGATIVE': '#505050',
                                             'NEUTRAL': '#A8A8A8',
                                             }
                         )
        fig_pie.update_layout(plot_bgcolor='#E8E8E8', paper_bgcolor='#E8E8E8', )



        return scatter_graph,fig_bar,fig_pie,
    else:
        return none_graph,none_graph,none_graph,


if __name__ == '__main__':
    app.run_server(debug = True)

