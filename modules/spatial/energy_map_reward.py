"""Directional energy-map reward for aligned WYZX FOA waveforms."""

import math

import torch


class EnergyMapReward:
    def __init__(self,device,angle_step_deg=6.,window_s=.5):
        if not math.isfinite(angle_step_deg) or not 0<angle_step_deg<=90 or not math.isclose(180/angle_step_deg,round(180/angle_step_deg),abs_tol=1e-8):
            raise ValueError('Energy angle step must divide 180 degrees')
        if not math.isfinite(window_s) or window_s<=0:raise ValueError('Invalid energy window')
        self.window_s=window_s
        self.height=round(180/angle_step_deg);self.width=2*self.height
        phi=(torch.arange(self.width,device=device,dtype=torch.float64)*angle_step_deg-180).flip(0)*math.pi/180
        nu=(torch.arange(self.height,device=device,dtype=torch.float64)*angle_step_deg-90)*math.pi/180
        n,p=torch.meshgrid(nu,phi,indexing='ij')
        self.matrix=torch.stack([torch.ones_like(p),math.sqrt(2)*n.cos()*p.sin(),n.sin(),math.sqrt(2)*n.cos()*p.cos()]).reshape(4,-1)
        left,right,weights=[],[],[]
        for row in range(self.height):
            size=max(1,round(self.width*math.sin(math.pi*(row+.5)/self.height)))
            for i in range(size):
                pos=i*(self.width-.5*(self.width/size))/size;lo=math.floor(pos)
                left.append(row*self.width+lo);right.append(row*self.width+min(lo+1,self.width-1));weights.append(pos-lo)
        self.left=torch.tensor(left,device=device,dtype=torch.long)
        self.right=torch.tensor(right,device=device,dtype=torch.long)
        self.weight=torch.tensor(weights,device=device,dtype=torch.float64)

    def map(self,audio):
        if audio.ndim!=2 or audio.shape[0]!=4 or audio.shape[-1]==0:raise ValueError('Energy map requires [4,N] WYZX')
        x=audio.double();moment=x@x.T/x.shape[-1]
        rms=(self.matrix*(moment@self.matrix)).sum(0).clamp_min(0).sqrt()
        rms=rms.reshape(self.height,self.width).flip(0).flatten()
        return rms[self.left]*(1-self.weight)+rms[self.right]*self.weight

    @staticmethod
    def valid(value):
        return bool((value-value.mean()).norm()>max(value.norm().item(),1e-12)*1e-8)

    def compare(self,reference,generated,sample_rate):
        if reference.shape!=generated.shape:raise ValueError('Energy reward requires aligned equal lengths')
        width=round(sample_rate*self.window_s)
        if width<1:raise ValueError('Energy window shorter than one sample')
        values=[];invalid_generated=0;total=0
        for start in range(0,reference.shape[-1],width):
            total+=1
            a=self.map(reference[:,start:start+width])
            if not self.valid(a):continue  # No defined spatial target; do not reward silence.
            b=self.map(generated[:,start:start+width])
            if not self.valid(b):
                values.append(a.new_tensor(-1.));invalid_generated+=1;continue
            ac,bc=a-a.mean(),b-b.mean()
            values.append((ac@bc/(ac.norm()*bc.norm())).clamp(-1,1))
        score=float(torch.stack(values).mean()) if values else 0.
        return score,bool(values),len(values)/max(total,1),invalid_generated/max(len(values),1)
