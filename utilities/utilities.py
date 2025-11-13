# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:34:12 2020

@author: jainakshay
"""
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import timedelta
# from parser_bincsv2df import bincsv_to_data
# from utilities import add_tsk_time2local
import pandas as pd
from datetime import datetime
import numpy as np
from os import listdir
from os.path import isfile,join
import os
from pathlib import Path
from os.path import sep
import subprocess
import csv
from matplotlib.backend_bases import KeyEvent
import io
import tempfile
import subprocess
from PIL import Image
import scipy.signal




# from zoneinfo import ZoneInfo
colors = plt.get_cmap('Set1').colors


def my_round(x, places=3):
    if isinstance(x, float):
        return round(x*(10**places))/(10**places)
    else:
        return [round(xx*(10**places))/(10**places) for xx in x]

def plot_with_time(time, data, ax=None, timeFormat="%H:%M:%S", title=None, ylim=None,timerange = None,  **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if timerange is not None:
        idx = (time >= timerange[0]) & (time <= timerange[1])
        time = time[idx]
        data = data[idx]
        
    if (isinstance(ax, np.ndarray)) | isinstance(ax, list):
        ax = ax[0]
    ax.plot(time,data,**kwargs)
    ax.xaxis.set_major_formatter(DateFormatter(timeFormat))    
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
#
# def copy_figure_to_clipboard_mac(fig):
#     # Save current zoom/pan view for each axis
#     ax_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
#     print(ax_limits)
#     for ax, (xlim, ylim) in zip(fig.axes, ax_limits):
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#     fig.canvas.draw_idle()
#     buf = io.BytesIO()
#     # Save using current view limits (manual restore below)
#     fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
#     buf.seek(0)
#     img = Image.open(buf)
#
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
#         img.save(f.name)
#         subprocess.run([
#             "osascript", "-e",
#             f'set the clipboard to (read (POSIX file "{f.name}") as «class PNGf»)'])

# def copy_figure_to_clipboard_mac(event):
#     canvas = event.canvas
#     width, height = canvas.get_width_height()
#
#     # Get raw RGBA bytes from GUI buffer
#     buf = canvas.buffer_rgba()
#     image = Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1)
#
#     # Save to a temporary PNG file
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
#         image.save(f.name)
#
#         # macOS clipboard copy
#         subprocess.run([
#             "osascript", "-e",
#             f'set the clipboard to (read (POSIX file "{f.name}") as «class PNGf»)'
#         ])
# from AppKit import NSPasteboard, NSPasteboardTypePNG, NSImage
# from Foundation import NSData
# def copy_figure_to_clipboard_mac(event):
#     canvas = event.canvas
#     buf = io.BytesIO()
#     canvas.figure.savefig(buf, format='png')
#     buf.seek(0)
#     data = NSData.dataWithBytes_length_(buf.read(), buf.getbuffer().nbytes)
#     image = NSImage.alloc().initWithData_(data)
#
#     pb = NSPasteboard.generalPasteboard()
#     pb.clearContents()
#     pb.setData_forType_(image.TIFFRepresentation(), NSPasteboardTypePNG)

def on_key(event):
    if event.key == 't':  # Press 't' to apply tight_layout
        event.canvas.figure.tight_layout()
        event.canvas.figure.canvas.draw()
    # elif event.key == 'c':
        # copy_figure_to_clipboard_mac(event)



def make_figure(name = None,nrows=1,ncols=1,sharex=True,sharey=False,figsize = None):
    # if figSize is not None:
    fig = plt.figure(name,figsize=figsize)
    fig.clf()
    axs = fig.subplots(nrows=nrows,ncols=ncols,sharex=sharex,sharey=sharey)
    fig.canvas.mpl_connect('key_press_event', on_key)

    if (nrows==1) & (ncols==1):
        return np.asarray([axs]),fig
    # Connect event
    return axs,fig

def remove_bookends_timeseries(df,start=1,end=1):
    df = df[(df.index > df.index[0]+timedelta(seconds=start)) & (df.index < (df.index[-1]-timedelta(seconds=end)))]
    return df
    


def my_hist(data,axs=None, title=None, bins=None,yaxis_type = 'absolute', clip_data = True, **kwargs):
    if axs is None:
        axs,fig = make_figure(name='Histogram')
    if (isinstance(axs, np.ndarray)):
        ax = axs[0]
    else:
        ax = axs
    if yaxis_type == 'percentage':
        weights = np.ones(len(data)) / len(data)
    else:
        weights = np.ones(len(data))
    if (clip_data) & (bins is not None):
        data = np.clip(data,min(bins),max(bins))
    ax.hist(data,bins=bins,weights=weights,**kwargs)
    ax.set_title(title)

def my_bar(data,axs=None, title=None, bins=None,yaxis_type = 'absolute', clip_data = True,showBarVal=True,edgecolor='k',align='edge',xlabel=None,ylabel=None, cummulative=False, **kwargs):
    if axs is None:
        axs,fig = make_figure(name='Bar Plot')
    if (isinstance(axs, np.ndarray)):
        ax = axs[0]
    else:
        ax = axs
    if yaxis_type == 'percentage':
        weights = np.ones(len(data)) / len(data)
    elif yaxis_type == 'pdf':
        weights = np.ones(len(data)) / (len(data)*(bins[1]-bins[0]))
    else:
        weights = np.ones(len(data))
    if (clip_data) & (bins is not None):
        data = np.clip(data,min(bins),max(bins))
    counts, labels = np.histogram(data,bins=bins,weights=weights)
    if cummulative:
        counts = np.cumsum(counts)
    bars = ax.bar(labels[:-1],counts,edgecolor=edgecolor,align=align,**kwargs)
    if showBarVal: 
        ax.bar_label(bars)
    ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)



def lat_long_semicircles_to_degree(semicircles):
    return (semicircles * (180 / 2**31))



def plot_dataframe(df,axs=None,timerange = None,color=colors[0],ylabels={},ylims={}, timeFormat="%H:%M:%S",title=None, **kwargs):
    cols = df.columns
    if timerange is not None:
        df = df[ (df.index >= timerange[0]) & (df.index <= timerange[1])]
    if axs is None:
        axs,fig = make_figure('Features',nrows=len(cols))
    for i,col in enumerate(cols):
        if title is None:
            plot_with_time(df.index, df[col],color=color,title=col,ax=axs[i],timeFormat=timeFormat,**kwargs)
        else:
            plot_with_time(df.index, df[col],color=color,title=title+' | '+col,ax=axs[i],timeFormat=timeFormat,**kwargs)
        if col in ylabels.keys():
            axs[i].set_ylabel(ylabels[col])

        if col in ylims.keys():
            axs[i].set_ylim(ylims[col])

def plot_dataframe_simple(df,axs=None,color=colors[0],ylabels={},ylims={},figName='Data Frame', **kwargs):
    cols = df.columns
    if axs is None:
        axs,fig = make_figure(figName,nrows=len(cols))
    for i,col in enumerate(cols):
        axs[i].plot(df.index, df[col],color=color,**kwargs)
        axs[i].set_title(col)
        axs[i].grid(True)
        if col in ylabels.keys():
            axs[i].set_ylabel(ylabels[col])
        if col in ylims.keys():
            axs[i].set_ylim(ylims[col])


def df_betweem_time(df,start_time,end_time,time_format='%Y-%m-%d %H:%M:%S',tzinfo=None):
    if isinstance(start_time,str):
        start_time = datetime.strptime(start_time,time_format)
        start_time = start_time.replace(tzinfo=tzinfo)
    if isinstance(end_time,str):
        end_time = datetime.strptime(end_time,time_format)
        end_time = end_time.replace(tzinfo=tzinfo)
        
    df2 = df[ (df.index >= start_time) & (df.index <= end_time)]
    return df2

def draw_line(ax,yVal, *args,**kwargs):
        xlim = ax.get_xlim()
        ax.plot(xlim,[yVal,yVal], *args,**kwargs)
        ax.set_xlim(xlim)

def draw_line_yaxis(ax,xVal, *args,**kwargs):
        ylim = ax.get_ylim()
        ax.plot([xVal,xVal],ylim, *args,**kwargs)
        ax.set_ylim(ylim)


def plot_bland_altman(x, y, ax=None, title="Bland-Altman Plot", units=""):
    """
    x, y: Arrays of paired measurements from two methods
    ax: Optional matplotlib axis to plot on
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate mean and difference
    mean = (x + y) / 2
    diff = y - x

    # Compute bias and limits of agreement
    bias = np.mean(diff)
    loa = 1.96 * np.std(diff)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(mean, diff, alpha=0.6,marker='.')
    ax.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    ax.axhline(bias + loa, color='gray', linestyle='--', label=f'+1.96 SD = {bias + loa:.2f}')
    ax.axhline(bias - loa, color='gray', linestyle='--', label=f'-1.96 SD = {bias - loa:.2f}')

    ax.set_xlabel(f'Mean of Methods ({units})')
    ax.set_ylabel(f'Difference (y - x) ({units})')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

def plotFourierTransform(signal, fs, window_size=None, overlap=None, ax=None, Nfft=None):
    """
    Compute the frequency and power spectrum of a 1D signal.
    If window_size and overlap are provided, use Welch's method.
    Otherwise, compute FFT of the whole signal.

    Parameters:
        signal (array-like): Input 1D signal.
        fs (float): Sampling frequency (Hz).
        window_size (int, optional): Window size for Welch's method.
        overlap (int, optional): Overlap between windows for Welch's method.

    Returns:
        f (np.ndarray): Array of sample frequencies.
        Pxx (np.ndarray): Power spectrum of the signal.
    """
    signal = np.asarray(signal)
    if window_size is not None and overlap is not None:
        f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=window_size, noverlap=overlap, nfft=Nfft)
    else:
        if Nfft is None:
            Nfft = len(signal)
        N = len(signal)
        f = np.fft.rfftfreq(Nfft, d=1/fs)
        fft_vals = np.fft.rfft(signal,n=Nfft)
        Pxx = np.abs(fft_vals) ** 2 / N
    if ax is None:
        axs,fig = make_figure('FFT',nrows=1,ncols=1)
        ax = axs[0]
    ax.stem(f, Pxx, basefmt="-", markerfmt='.')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum')
    ax.grid(True)

    # return f, Pxx