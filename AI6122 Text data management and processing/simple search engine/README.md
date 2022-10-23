# Environment Building

Platform information: GNU/Linux `Ubuntu 20.04 LTS` on `5.10.102.1-microsoft-standard-WSL2` core, x86 64 bit platform

Python version: 3.10.4

There should be no non-built-in packages needed except `JCC` and `PyLucene`, you can refer to `environment.txt` for all packages of the environment if some goes wrong.

## Building PyLucene for the environment

The `PyLucene` works by building a python module with `JCC` to implant a Java virtual machine into Python. You can find the package here: [Apache Lucene - Welcome to PyLucene](https://lucene.apache.org/pylucene/index.html)

Please follow the following procedure to build and install PyLucene for your environment. The whole procedure should take about 2 hours.

**All paths appearing in the code blocks are my case, you should change the paths accroading to your case.**

* Prepare a python environment, it can be a naked python directly installed or an environment created with `Anaconda` or `virtualenv`. For `Anaconda` users, you can use: `conda create -n ntu_text -python=3.10`

* Prepare `GCC` toolkit. For most Linux OS, GCC toolkits should be built-in.

* Prepare `JDK` toolkit. For `apt` users, run: `sudo apt-get install openjdk-11-jdk-headless`

* Find true position of `JDK` executable files. You can use `which javac` command and `ls -l` command to find the folder through symbol links. In my case, it is `/usr/lib/jvm/java-11-openjdk-amd64/bin`.

* Install `ANT`, a tool provided by Apache, used to automatically build java projects:
  
  ```bash
  mkdir ~/temp
  cd ~/temp
  wget https://dlcdn.apache.org//ant/binaries/apache-ant-1.10.12-bin.tar.gz
  tar zxvf apache-ant-1.10.12-bin.tar.gz
  mv ~/temp/apache-ant-1.10.12 ~/apache-ant-1.10.12
  ```
  
  Add following content to `~/bashrc`:
  
  ```bash
  export ANT_HOME="$HOME/apache-ant-1.10.12"
  export PATH="$PATH:$ANT_HOME/bin"
  export ANT_OPTS="-Xms1300m -Xmx2048m -XX:PermSize=128M -XX:MaxNewSize=256m -XX:MaxPermSize=256m"
  ```

* Download PyLucene:
  
  ```bash
  cd ~/temp
  wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.1.0-src.tar.gz
  tar zxvf pylucene-9.1.0-src.tar.gz
  mv ~/temp/pylucene-9.1.0 ~/pylucene-9.1.0
  ```

* Insall `JCC`: change to your download position of pylucene, `~/pylucene-9.1.0` in my case, open `setup.py`, find a dictionary called `JDK`, change the value of item `linux` to your java position, `/usr/lib/jvm/java-11-openjdk-amd64` in my case.

* **Enter the environment you want to install PyLucene for**, all `python` command in following instructions refer to the `python` in this environment, and note the position of `python` executable file, `~/anaconda3/envs/ntu_text/bin/python` in my case.

* Run:
  
  ```bash
  cd ~/pylucene-9.1.0
  python setup.py build
  sudo ~/anaconda3/envs/ntu_text/bin/python setup.py install
  ```
  
  * **If you want to build `PyLucene` for multiple environments, backup the `pylucene-9.1.0` folder now.**

* Insatll `PyLucene`
  
  * Open `~/pylucene-9.1.0/Makefile`, add the following contents before the line `# You need to uncomment and edit the variables below in the section`. This is merely an example, **you should read the comments and follow your case**. **Note the path here MUST be ABSOLUTE PATH**
    
    ```bash
    PREFIX_PYTHON=~/anaconda3/envs/ntu_text
    ANT=~/apache-ant-1.10.12/bin/ant
    PYTHON=$(PREFIX_PYTHON)/bin/python
    JCC=$(PYTHON) -m jcc --shared
    NUM_FILES=16
    ```
  
  * Run `python -m jcc` in the environment, if a help file for `JCC` is given, then `JCC` has been installed correctly.
  
  * Run `make` in `~/pylucene-9.1.0`.
    
    * If you encountered in time out problem, try again or follow the following instructions:
    
    * Open `~pylucene-9.1.0/lucene-java-9.1.0/gradle/wrapper/gradle-wrapper.properties` file, find the `distributionUrl` item, download the file specified by this url, and find the path specified by the `zipStorePath` item, find a folder with a name like `gradle-some version-something`, in which find a folder with unreadable name and `.lck` and `.ok` file inside.
    
    * Put the zip file you just downloaded in, go back to `~/pylucene-9.1.0` and run `make` again.
  
  * Run `sudo make install`

Now you should have successfully installed PyLucene, try this code in python:

```python
import lucene
lucene.initVM()
```

If no error has been raised, then you have successfully installed PyLucene.

# How To Use

Run `python buildIndex.py` to build index. Change the `original_data_paths` and the `default_index_path` to specify the data files and where to write indices.

Run `python searcher.py` to search. Follow the instructions given by the program to specify the index path and max retrieve count.

# Examples

```
Welcome to ReviewSearcher.searcher.
Please specify index path, press Enter to use default

Please specify how many files to retrieve at max, press Enter to use default 50.
1
Initializing indices, you can read about the advanced search grammer now:
Query grammar: 
    Query  ::= ( Clause )*
    Clause ::= ["+", "-"] [<TERM> ":"] ( <TERM> | "(" Query ")" )
eg.: `+field:keyword1 keyword2 -field:"a phrase"` 
Features supported:
    wildcard query(`?` and `*`) is possible, just should not be at the start of a word
    regexp query: `/RegExp/`
    fuzzy query: `fuzzy~2`, default edit distance is 2, modifiable
    numeric query: for exact query, use it like it's a keyword;
        for range query, use this: `field:{start TO end]`,
        where `{}` means exclusive, `[]` means inclusive,
Fields supported:
    "reviewerID": the id of the reviewer, identical match only
    "asin": the id of the product, identical match only
    "reviewerName": the name of the reviewer
    "reviewText": the review given by the reviewer
    "overall": the rate given by the reviewer, RegEx`([0-9]\.[0-9])`
    "summary": a simple summary of the review
    "unixReviewTime": the unix time stamp of the review time, RegEx`([0-9]{10})`
With this program, you can also add `AND` or `OR` at the start of the query, to specify the boolean operators
Please note that the "overall" and "unixReviewTime" field are stored as strings and thus the number of digits must correpond
You can put "NUM" as a query to invoke range search methods with numbers.
Similarly, type `HELP` can bring this tutorial to you again.
Default operator is `AND`, and default field is "reviewText".
You can use Crtl+C to exit anytime.

eg.: `OR "Kit Kat" chocolate `
eg.: `chocolate cheap overall:{3.0 TO *}`

Initializing completed
Please give your query:"text processing"
1 docs have been retrieved in 0:00:00.055244 time.
The No.1 hit, score: 6.388261795043945, doc id: 729082
-------
<reviewerID>: A221HTD29MIAKU
<asin>: B002TLTH7K
<reviewerName>: Visual8
<reviewText>: I'm not only an Apple advanced user, but also an Apple fan for all of their products. This mouse is excellent for everything (design, web browsing, text processing). Has a lot of functions and buttons, but the only thing is that the trackball sometimes stops working because of the dirt.
<overall>: 4.0
<summary>: Great mouse.
<unixReviewTime>: 1342396800
-------
Please give your query:iajsodiffna
0 docs have been retrieved in 0:00:00.009520 time
Please give your query:(chocolate OR "Kit Kat") (cheap OR inexpensive) overall:[4.0 TO 6.0]
1 docs have been retrieved in 0:00:00.136154 time.
The No.1 hit, score: 8.00647258758545, doc id: 1822488
-------
<reviewerID>: A1RMAFZ1RS0P5F
<asin>: B00GN648WG
<reviewerName>: Axon
<reviewText>: The NV Tegra 4 sure makes everything run pretty snappy on the tablet. The two speakers makes the sound great, watching videos or listening to music. The screen is nice even though its only 1280x800(736) still its good enough for a 7in tablet, it plays most common formats mp4 and some avi. The feel of the device is quite nice, well built. It doesn't feel cheap. The camera is also nice, despite its only 5mp(no flash), it has auto focus and a set of potions to play around with it. The stylus pen its rather good, surprisingly it feels heavy for a pen. I might order one for my slate 7 plus. The only 2 and half cons are, its still running on JB 4.2 no word on HP if it will ever release 4.3 or move to Kit Kat, kind of strange given this is a rebranded Tegra note and that one NV already release the update to Kit Kat, so why won't HP do the same? The other issue you can't move apps to the micro SD card, on the plus side it allows micro SD cards up to 64GB just reformat it using FAT32 before insert into the tablet, feel free to add movies, photos and songs but that's it. The other odd issue its that some games are not compatible with this device, even though the games were meant to be played on the Tegra 4. Overall its a very nice tablet with lots of features, forgot to mention I haven't try the mini HDMI connector to plug the tablet to my tv, so if anyone has try it please let me know how well it works.
<overall>: 5.0
<summary>: Nice update from the Slate 7 plus
<unixReviewTime>: 1396310400
-------
Please give your query:
Thanks for using :)
```

## Explanation

First, when asked index path, press enter to use default path, `index`.

Then, specify only 1 document to retrieve

First query is `"text processing"`, the result is a review to an Apple mouse

Second query is `iajsodiffna`, a piece of nonsense code, no result has been retrieved.

Third query is the example query shown in report, `(chocolate OR "Kit Kat") (cheap OR inexpensive) overall:[4.0 TO 6.0]`, the result is a review to a tablet. As explained in the report, this report has a highest score because it mentioned many times about Android 4, whose code name is "Kit Kat".

Then we use empty query to exit.
