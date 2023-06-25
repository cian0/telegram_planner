import os
import json
import traceback
import time
import boto3

import openai
from loguru import logger
from chalice import Chalice
from telegram.ext import (
    Dispatcher,
    MessageHandler,
    Filters
)

from telegram import ParseMode, Update, Bot, MessageEntity, ChatAction
from chalice.app import Rate

from chalicelib.utils import generate_transcription, send_typing_action

from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory

from langchain.chat_models import ChatOpenAI
import pinecone

# setup keys and tokens 
TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Setup Chalice Lambda app
APP_NAME = "chatgpt-telegram-bot"
MESSAGE_HANDLER_LAMBDA = "message-handler-lambda"

app = Chalice(app_name=APP_NAME)
app.debug = True

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Conversations')

# Telegram bot
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

# Initialize Embedding Model
model_name = 'text-embedding-ada-002'
logger.info("initializing")
embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
logger.info("OpenAIEmbeddings initialized")

# Initialize the Pinecone vector database
index_name = 'youtube-retrieval-augmentation'
pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp-free")
logger.info("pinecone initialized")

# Switch back to the normal index for LangChain
index = pinecone.Index(index_name)
vectorstore = Pinecone(index, embed.embed_query, "text")

logger.info("vectorstore initiated")
# Initialize the generative question-answering system
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    # model_name='gpt-3.5-turbo',
    model_name='gpt-3.5-turbo-16k',
    # model_name='gpt-4',
    # temperature=0.7,
    temperature=0,
)
logger.info("ChatOpenAI initiated")


obj = {
    "qa": None,
    "memory": None,
    "session_id": None
}

logger.info("ConversationalRetrievalChain initiated")

#####################
# Telegram Handlers #
#####################


def ask_chatgpt(text, chat_id, qa, memory):
    chat_history = memory.chat_memory.messages#[]#get_conversation(chat_id)
    
    query = '''
    
    You will play the role of a startup advisor who is emulating the capabilities of the best startup advisors, 
    you have access to a wealth of knowledge about startups from well known incubators and accelerators, 
    and must treat me as a mentee who is asking you for advise. 
    Your response format should focus on the context you have access with and how it pertains to the mentee's questions 
    and asking clarifying questions. 
    Always speak in first person and take whatever the context or speaker says as your own idea/opinion.

    As a startup advisor, your thought process when a mentee asks you a question should be guided by a few key principles:

    Understand the Context: Before providing any advice, it's crucial to understand the context of the question. This includes understanding the current stage of the startup, the specific challenges they're facing, and the goals they're trying to achieve.

    Leverage Your Experience and Expertise: As an advisor, your primary value comes from your experience and expertise. When answering a question, think about how your past experiences can provide insight into the situation.

    Think Strategically: Rather than just addressing the immediate question, think about the bigger picture. How does this question fit into the overall strategy of the startup? What long-term implications could your advice have?

    Promote Learning and Independence: Your goal as an advisor isn't to provide all the answers, but to help the mentee develop their own problem-solving skills. Instead of just giving an answer, consider guiding the mentee to find the answer themselves. This could involve asking probing questions, suggesting resources for further research, or providing a framework for decision-making.

    Be Honest and Constructive: If you don't know the answer to a question, it's better to admit it than to give potentially misleading advice. Similarly, if you think the mentee is heading in the wrong direction, it's important to provide constructive feedback, even if it might be hard to hear.

    Consider Ethical Implications: Finally, always consider the ethical implications of your advice. Ensure that your advice aligns with ethical business practices and promotes a culture of integrity within the startup.

    Remember, as an advisor, your role is to guide and support the mentee, not to make decisions for them. Your thought process should be focused on providing the best possible guidance to help the mentee navigate their own startup journey.

    The mentee's first message:
    ''' + text
    if chat_history:
        query = text
    result = qa({"question": query})
    logger.info(query)
    logger.info(result['answer'])
    return result['answer']

@send_typing_action
def process_voice_message(update, context):
    # Get the voice message from the update object
    voice_message = update.message.voice
    # Get the file ID of the voice message
    file_id = voice_message.file_id
    # Use the file ID to get the voice message file from Telegram
    file = bot.get_file(file_id)
    # Download the voice message file
    transcript_msg = generate_transcription(file)
    message = ask_chatgpt(transcript_msg, chat_id)

    chat_id = update.message.chat_id
    context.bot.send_message(
        chat_id=chat_id,
        text=message,
        parse_mode=ParseMode.MARKDOWN,
    )

def get_conversation(chat_id):
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('chat_id').eq(str(chat_id)),
        ScanIndexForward=True  # Sort by timestamp in ascending order
    )

    # Extract the user_message and bot_message from each item and store them as tuples in an array
    conversation = [(item['user_message'], item['bot_message']) for item in response['Items']]

    return conversation

def save_conversation(chat_id, user_message, bot_message, memory):
    timestamp = int(time.time())
    table.put_item(
        Item={
            'chat_id': str(chat_id),  # Convert chat_id to string
            'timestamp': timestamp,
            'user_message': user_message,
            'bot_message': bot_message
        }
    )

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(bot_message)


# @send_typing_action
def process_message(update, context):
    chat_id = update.message.chat_id
    chat_text = update.message.text

    message = update.message
    chat_id = message.chat_id
    text = message.text

    if message.reply_to_message:
        replied_message = message.reply_to_message
        replied_text = replied_message.text

        # Check if the reply is in a group chat
        if message.chat.type != "private":
            # Check if the replied message was sent by the bot
            if replied_message.from_user.id == 6194891324:
                reply_text = "You replied to my message in a group chat: " + replied_text
            else:
                reply_text = "You replied to a message from someone else in a group chat: " + replied_text
                return
        else:
            reply_text = "You replied to my direct message: " + replied_text

    else:
        # Check if the message is from a private chat
        if message.chat.type == "private":
            reply_text = "You messaged me directly: " + text
        else:
            mentioned_bot = False
            if update.message.entities:
                
                for entity in update.message.entities:
                # Check for both regular mentions and text mentions
                    if entity.type == MessageEntity.MENTION:
                        start = entity.offset
                        end = start + entity.length
                        mention = update.message.text[start:end]

                        if  'chatgpt_startup_bot' in mention[1:]:
                            logger.info("The bot was mentioned!")
                            mentioned_bot = True
                            break
                        
                    elif entity.type == MessageEntity.TEXT_MENTION:
                        user = entity.user
                        if  'chatgpt_startup_bot' in user.username:
                            logger.info("The bot was mentioned!")
                            mentioned_bot = True
                            break
                    
            if not mentioned_bot:
                return
            
            reply_text = "You messaged in a group chat: " + text
    
    logger.info(reply_text)

    if obj["qa"] is None or obj["session_id"] != chat_id:
        message_history = DynamoDBChatMessageHistory(
                table_name="startup_bot_chat_history", session_id=chat_id
        )

        memory = ConversationBufferMemory(
                memory_key="chat_history", chat_memory=message_history, return_messages=True
        )
        logger.info("ConversationBufferMemory initiated")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
        obj["qa"] = qa
        obj["memory"] = memory
        obj["session_id"] = chat_id

    if "/_reset" in chat_text:
        reset(update, context, obj["memory"])
        return

    try:
        qa = obj["qa"]
        memory = obj["memory"]
        context.bot.send_chat_action(chat_id, action=ChatAction.TYPING)
        message = ask_chatgpt(chat_text, chat_id, qa, memory)
    except Exception as e:
        app.log.error(e)
        app.log.error(traceback.format_exc())
        context.bot.send_message(
            chat_id=chat_id,
            text="There was an error handling your message :(",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        context.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN,
        )
        save_conversation(chat_id, chat_text, message, memory)

def delete_conversation(chat_id):
    # Get all items for the chat_id
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('chat_id').eq(str(chat_id))
    )

    # Delete each item
    for item in response['Items']:
        table.delete_item(
            Key={
                'chat_id': str(chat_id),
                'timestamp': item['timestamp']
            }
        )

def reset(update, context, memory):
    chat_id = update.message.chat_id
    # delete_conversation(chat_id)

    memory.chat_memory.clear()

    context.bot.send_message(
        chat_id=chat_id,
        text="Chat history has been reset.",
        parse_mode=ParseMode.MARKDOWN,
    )


############################
# Lambda Handler functions #
############################

@app.lambda_function(name=MESSAGE_HANDLER_LAMBDA)
def message_handler(event, context):

    # dispatcher.add_handler(CommandHandler("reset", reset))
    dispatcher.add_handler(MessageHandler(Filters.text, process_message))
    dispatcher.add_handler(MessageHandler(Filters.voice, process_voice_message))

    try:
        dispatcher.process_update(Update.de_json(json.loads(event["body"]), bot))
    except Exception as e:
        logger.info(e)
        return {"statusCode": 500}

    return {"statusCode": 200}
