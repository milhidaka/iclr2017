# -*- coding:utf-8 -*-

"""
Access variable server data without HTTP (local machine)
"""

import os
import sys
import sqlite3
import varserver_config

STATUS_RESERVED = 0
STATUS_WRITING = 1
STATUS_COMPLETE = 2
STATUS_DELETED = 3

def init_db(path):
    db = sqlite3.connect(path)
    cur = db.cursor()
    cur.execute("CREATE TABLE objs(id integer primary key, created_at text not null default CURRENT_TIMESTAMP, size integer not null, status integer not null, name text)")

def open_db():
    db_path = os.path.join(varserver_config.DATA_DIR, "vs.db")
    if not os.path.exists(varserver_config.DATA_DIR):
        os.mkdir(varserver_config.DATA_DIR)
    if not os.path.exists(db_path):
        init_db(db_path)
    db = sqlite3.connect(db_path)
    return db

def make_path(blob_id):
    return os.path.join(varserver_config.DATA_DIR, "{}.bin".format(blob_id))

def reserve_var(db, name=None):
    cur = db.cursor()
    cur.execute("INSERT INTO objs(size, status, name) VALUES(?, ?, ?)", (0, STATUS_RESERVED, name))
    blob_id = cur.lastrowid
    db.commit()
    return blob_id

def write_var(db, blob_id, body):
    """
    """
    size = len(body)
    with open(make_path(blob_id), "wb") as f:
        f.write(body)
    cur = db.cursor()
    cur.execute("UPDATE objs SET size=?, status=? WHERE id=?", (size, STATUS_COMPLETE, blob_id))
    db.commit()

def reserve_write_var(db, body, name=None):
    blob_id = reserve_var(db, name)
    write_var(db, blob_id, body)
    return blob_id

def read_var(db, blob_id):
    with open(make_path(blob_id), "rb") as f:
        body = f.read()
    return body

def remove_var(db, blob_id):
    cur = db.cursor()
    cur.execute("UPDATE objs SET status=? WHERE id=?", (STATUS_DELETED, blob_id))
    db.commit()
    try:
        os.remove(make_path(blob_id))
    except:
        pass
