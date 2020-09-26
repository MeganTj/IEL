# IEL

A codebase for IEL bots and simulation analysis software. 

## Implemented features 



## Running IEL Simulations: Last updated on 9/20/2020


Purpose: This is a step-by-step guide on how to run simulations on Flexemarkets (https://flexemarkets.com/site), using the bot_launcher.py script (referred to as bot launcher or launcher from here on) as well as an IEL manager and IEL trader bots. The tutorial is split into two parts: the first is an overview of the Flexemarkets platform, and the second covers the necessary modifications to the bot Python scripts and an explanation of the bot launcher.

### Part 1: Setup FM account

Consult the FM documentation here: https://docs.google.com/document/d/1P8NnPDw7ScGzjhe_OIakpVP83FhhwZRKpkkYZnUCsas/edit and go through all the steps listed before proceeding to Part 2. If one needs help getting an account more quickly, email peter.bossaerts@unimelb.edu.au or jan.nielsen@adhocmarkets.com. They will provide the username for logging in, though the password is set by the user and not stored by FM.

The difference between Adhocmarkets and Flexemarkets is that as of the time this guide is written, only Flexemarkets has the capability for simulations to be run using manager bots, so Flexemarkets is preferred.  

When creating a market, be sure to remember the item name (this is also listed when the market is open, above the order book) as the exact item name (which is sensitive to case) must be the name of the "MARKET_1" variable in the IEL manager. The Marketplace ID for each market is displayed on the list of markets following creation. Besides filling in the market name, description, and item name, it is recommended to keep all other settings as the default. 

When running simulations with a manager, the initial allocation template will not matter, since the manager determines initial holdings before every period. The traders that are added to the market will correspond with either human users or trader bots (each bot has its own email and password, which is used in Part 2).     


### Part 2: Running simulations from the command line (9/20/20 version)

IMPORTANT: Make sure the market is closed before running a simulation.

1) Set the "CONFIG_FILE" variable in bot_launcher to the absolute path of the Excel configuration file. For example, `CONFIG_FILE = "/Users/MeganT/Documents/mytest/IEL_config.xlsx"`

2) Instructions on filling out the configuration file are as follows. 

Row 1 of sheet 1 of the configuration file is the header for the overall parameters that all bot share (account name, market id, market item, account email, account password), and these values should be updated in row 2. The header for parameters that can vary per-bot is row 4, and the parameters for each bot with bot number n should be updated in the n'th row following row 4. In sheet 2, the valuation sets are listed as columns that are named in row 1, and one of these names must be exactly referenced in the "val_set" parameter for each bot. The valuation sets start with the value of item 0 at row 2, the value of item 1 at row 3, and so on. 

For brevity, the configuration file assumes a naming convention. Have the number of bots to be created = N. Let x be the bot’s number. Then, the program name is IELvY_bx where Y is the version number.
```
bot_name = testx@test  
bot_password= passx.
name = “IELx”
```

All overall parameters are required, while only "val_set" for bots is required. If min_holdings/max_holdings are left blank or are invalid (< 0 for min_holdings or > len(values) - 1 for max_holdings), the default holdings value (0 for min_holdings, len(values) - 1 for max_holdings) will be used. Note that it isn't required to fill out the "bot_num" column -- this is included in the template for readability. Cells for any optional parameters should be left blank to use the default value for that parameter. 

Other optional parameters are

* J -  The number of items in "considered set" 
* muv - The rate of mutation of value
* sigmav - The variance on the mutation of value. 
* mul - The rate of mutation of length
* K - The max length of a "considered strategy"
* T - A measure of how far back to start keeping track of past prices



### Part 2: Running simulations from the command line (8/8/20 version)

IMPORTANT: Make sure the market is closed before running a simulation. 

1) Within the IEL manager Python script, make sure to update `MARKET_1` with the item name that is chosen when creating a market on FM in Part 1. Also make sure to update `MARKETPLACE_ID`, `FM_ACCOUNT`, `FM_EMAIL`, and `FM_PASSWORD` with the appropriate information.

One needs to change the `SAVE_TO` variable in the IEL manager script to the absolute file path of an already-existing directory on the local machine. Simulation results in the format of individual csv files for each simulation period are saved to this directory. For example, on a Mac OS, one can set

`SAVE_TO="/Users/MeganT/Documents/IEL/"`

Note: On other operating systems, the format of the file path may be different. On Windows, the absolute file path begins with `C:`, and backslashes instead of forward slashes are used.

The forward slash (or backslash) at the end of the file path is required. When run, the IEL manager will create a new folder, i.e. `/Users/MeganT/Documents/IEL/sim1/`, in which to save csv files. Separate folders (named sim1, sim2, etc.) are created for each simulation, in case one is running the same simulation multiple times with the bot launcher.

2) Before moving on to running the bot launcher, make sure that each individual trader has its corresponding Python script. One can simply make copies of a template file, and for each trader, modify the code within the `if __name__ == '__main__'` block as follows. The email and password in the IELAgent constructor should correspond to a unique Flexemarket user that is imported through the Flexemarkets website (done in Part 1). The constructor of the trader takes in a valuation array, consisting of how much a trader values items 0 to n (where the valuation of item 0 is always 0 -- this is only needed since arrays are 0-indexed). The market id should remain the same across all agents so that all bots trade on the same market. 

3) It is assumed that one is running the bot launcher on the command line, in the same directory as all the bot scripts (of both the manager and traders), since the launcher takes in directives for running simulations through command line arguments. Also, it is assumed that the trader bot scripts follow this naming convention: prefix + number, where the prefix is the same for all traders and the number varies from 1 to n, where n is the total number of traders to run. For example, if the prefix is "IELv3_b" and n = 3, running the following command:

`python bot_launcher.py IELv3_b 3`

will run IELv3_b1.py, IELv3_b2.py, and IELv3_b3.py simultaneously. 

Note: Without an IEL manager, one needs to manually download the simulation data from the user interface, and this data will contain the orders and holdings from all past periods run on a given market.

4) Running an IEL manager along with a manager simply involves including the `-m` optional argument, followed by the full name of the IELmanager file without the `.py` extension. Running the following command:

`python bot_launcher.py IELv3_b 2 -m IELmanager1`

makes sure to run IELmanager1.py before IELv3_b1.py, IELv3_b2.py, and IELv3_b3.py. The results of the simulation are stored in the "sim1" folder within the SAVE_TO directory.

5) Assuming that simulations are run with an IEL manager, one can repeat the same simulation x times by including the "-r" optional argument, followed by x, which must be an integer. A new simulation starts and ends when the IEL manager opens the market for the first period and then closes the market after the final period. Running the following command:

`python bot_launcher.py IELv3_b 2 -m IELmanager1 -r 2`

repeats the process in 3) twice. The results of the first and second runs are stored in "sim1" and "sim2" respectively within the SAVE_TO directory.

6) To stop a simulation before it ends, CTRL+Z OR CTRL+C on a Unix command line will terminate all bots cleanly. 


