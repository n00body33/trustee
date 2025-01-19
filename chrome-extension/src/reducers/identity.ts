import { Identity } from '@semaphore-protocol/identity';
import { sha256 } from '../utils/misc';
import { Dispatch, SetStateAction, useEffect, useState } from 'react';

export class IdentityManager {
  async getIdentity(): Promise<Identity> {
    const identityStorageId = await sha256('identity');
    try {
      const storage = await chrome.storage.sync.get(identityStorageId);
      const identity = storage[identityStorageId];
      if (!identity) {
        return this._createIdentity();
      }
      return new Identity(identity);
    } catch (e) {
      return this._createIdentity();
    }
  }

  async _saveIdentity(identity: Identity): Promise<void> {
    const identityStorageId = await sha256('identity');
    try {
      await chrome.storage.sync.set({
        [identityStorageId]: identity.privateKey.toString(), // Only PRIVATE KEY is enough to reconstruct the identity
      });
    } catch (e) {
      console.error('Error saving identity', e);
    }
  }

  async _createIdentity(): Promise<Identity> {
    console.log('creating identity');
    const identity = new Identity();
    await this._saveIdentity(identity);
    return identity;
  }

  async loadIdentity(privateKey: string): Promise<Identity> {
    const identity = new Identity(privateKey);
    await this._saveIdentity(identity);
    return identity;
  }
}

export const useIdentity = (): [
  Identity | null,
  Dispatch<SetStateAction<Identity | null>>,
] => {
  const [identity, setIdentity] = useState<Identity | null>(null);
  useEffect(() => {
    (async () => {
      const identityManager = new IdentityManager();
      const identity = await identityManager.getIdentity();
      setIdentity(identity);
    })();
  }, []);
  return [identity, setIdentity];
};
