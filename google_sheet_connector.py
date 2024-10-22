import pygsheets

client = pygsheets.authorize(service_account_file="chatbot-responses-3fbcd29275d3.json") 
#debugging purpose

spreadsht = client.open("chatbot responses") 
worksheet=spreadsht.worksheet_by_title("Sheet1")
def google_sheet_connector(inp,op,time):
 worksheet.append_table(values=[inp, op, time]) 
# print("code works successfully")
# next_row = len(worksheet.get_all_values()) + 1  # gets the last row with data and adds 1
# print(next_row)
# Insert data at the next available row
# worksheet.insert_rows(next_row, values=data, inherit=False)
# worksheet.update_values('A1', data) 
# opens a spreadsheet by its name/title 
# spreadsht = client.open("chatbot responses")

