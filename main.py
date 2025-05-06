import pandas as pd

from common.utils import BIOPSY_TYPE


def main():
    print("Hello from graph-xpci!")
    print("Available biopsy types:")
    for biopsy in BIOPSY_TYPE:
        print(f"- {biopsy}")


if __name__ == "__main__":

    # main()
    # df = pd.read_csv('data/fresh/test.csv')
    # df = df[df.mask_infiltrates.notna()]
    # for img in df.image:
    #     img = img.split('/')[-1]
    #     img = img[:-4]
    #     print(img, end=' ')
    # print()
    # print(df.image)

    # import os
    # from tqdm import tqdm

    # df = pd.DataFrame()
    # dt = {'gen':1, 'gen2':1, 'real0R':0}
    # fol = '/storage/home/dvrsnak'
    # for k, v in dt.items():
    #     for i in tqdm(os.listdir(os.path.join(fol, k))):
    #         if i.endswith('.png'):
    #             df = pd.concat([df, pd.DataFrame({'image': [os.path.join(k,i)], 'target': [v], 'biopsy': [''], 'type':['gen' if 'gen' in k else 'fresh'], 'original_image': ['' if 'gen' in k else i.split('/')[-1][:-17]]})])
    # print(df.count())
    # df.to_csv(fol+'/gen.csv', index=False)

    import os

    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('/storage/home/dvrsnak/gen_csv/train.csv')
    print(df.groupby(df.target).count())
    # df = pd.read_csv('/storage/home/dvrsnak/gen.csv')
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    # os.makedirs('/storage/home/dvrsnak/gen_csv', exist_ok=True)
    # train_df.to_csv('/storage/home/dvrsnak/gen_csv/train.csv', index=False)
    # test_df.to_csv('/storage/home/dvrsnak/gen_csv/test.csv', index=False)
    # print(train_df.count())
    # print(test_df.count())